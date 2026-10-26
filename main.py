from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import json
import vector_db
import os
from dotenv import load_dotenv
from langfuse import Langfuse

# Загружаем ключи из .env
load_dotenv()

# Инициализируем Langfuse (SDK v4)
langfuse = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_BASE_URL", "https://cloud.langfuse.com")
)

app = FastAPI(title="Enterprise RAG Service")

class ChatRequest(BaseModel):
    message: str

class KnowledgeAdd(BaseModel):
    text: str

@app.post("/add_knowledge")
def add_knowledge(data: KnowledgeAdd):
    vector_db.add_document(data.text)
    return {"status": "ok", "length": len(data.text)}

async def stream_rag_response(user_message: str):
    # Создаем корневой Span (Trace) для всего запроса
    with langfuse.start_as_current_observation(
        as_type="span", 
        name="chat_request", 
        input=user_message
    ) as trace_span:
        
        try:
            # 1. Вложенный Span для RAG-поиска
            with langfuse.start_as_current_observation(
                as_type="span", 
                name="rag_retrieval"
            ) as retrieval_span:
                context = vector_db.search_docs(user_message)
                retrieval_span.update(output=context)
            
            system_prompt = (
                "Ты корпоративный ассистент. Используй контекст ниже для ответа.\n"
                f"КОНТЕКСТ:\n{context}"
            )
            
            # 2. Вложенная Generation для вызова LLM
            with langfuse.start_as_current_observation(
                as_type="generation",
                name="ollama_generation",
                model="qwen3.5:2b",
                input=user_message,
                metadata={"context": context}
            ) as generation:
                
                client = ollama.AsyncClient(host='http://localhost:11434')
                stream = await client.chat(
                    model='qwen3.5:2b',
                    messages=[
                        {'role': 'system', 'content': system_prompt},
                        {'role': 'user', 'content': user_message}
                    ],
                    stream=True
                )
                
                full_response = ""
                async for chunk in stream:
                    content = chunk['message']['content']
                    if content:
                        full_response += content
                        yield f"data: {json.dumps({'text': content})}\n\n"
                        
                # Сохраняем итоговый ответ в Generation
                generation.update(output=full_response)
            
            # Обновляем корневой Span итоговым ответом
            trace_span.update(output=full_response)
            
            yield "data: [DONE]\n\n"
            
        except Exception as e:
            error_msg = f"Ошибка генерации: {str(e)}"
            trace_span.update(level="ERROR", status_message=error_msg)
            yield f"data: {json.dumps({'text': error_msg})}\n\n"
            yield "data: [DONE]\n\n"

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        stream_rag_response(req.message), 
        media_type="text/event-stream"
    )
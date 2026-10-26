from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import json
import vector_db

app = FastAPI(title="Enterprise RAG Service")

class ChatRequest(BaseModel):
    message: str

class KnowledgeAdd(BaseModel):
    text: str

@app.post("/add_knowledge")
def add_knowledge(data: KnowledgeAdd):
    vector_db.add_document(data.text)
    return {"status": "ok", "length": len(data.text)}

# Меняем функцию на асинхронную
async def stream_rag_response(user_message: str):
    # Используем AsyncClient для неблокирующего стриминга
    client = ollama.AsyncClient(host='http://localhost:11434')
    
    try:
        # В реальном проекте search_docs тоже лучше сделать асинхронным, 
        # но для локальной Qdrant это занимает миллисекунды.
        context = vector_db.search_docs(user_message)
        
        system_prompt = (
            "Ты корпоративный ассистент. Используй контекст ниже для ответа.\n"
            f"КОНТЕКСТ:\n{context}"
        )
        
        # Ждем установки соединения (await)
        stream = await client.chat(
            model='qwen3.5:2b',
            messages=[
                {'role': 'system', 'content': system_prompt},
                {'role': 'user', 'content': user_message}
            ],
            stream=True
        )
        
        # Читаем поток асинхронно
        async for chunk in stream:
            content = chunk['message']['content']
            if content:
                yield f"data: {json.dumps({'text': content})}\n\n"
                
        yield "data: [DONE]\n\n"
        
    except Exception as e:
        # Если произойдет ошибка, мы не оборвем соединение, а отправим текст ошибки
        error_msg = f"Ошибка генерации: {str(e)}"
        yield f"data: {json.dumps({'text': error_msg})}\n\n"
        yield "data: [DONE]\n\n"

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        stream_rag_response(req.message), 
        media_type="text/event-stream"
    )
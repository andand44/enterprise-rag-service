from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import ollama
import json

app = FastAPI(title="Enterprise RAG Service")

class ChatRequest(BaseModel):
    message: str

# Вспомогательная функция-генератор для стриминга
def stream_ollama_response(user_message: str):
    client = ollama.Client(host='http://host.docker.internal:11434')
    
    # В реальном проекте тут был бы поиск в Qdrant (RAG)
    system_prompt = "Ты полезный ассистент. Отвечай кратко."
    
    stream = client.chat(
        model='qwen3.5:2b',
        messages=[
            {'role': 'system', 'content': system_prompt},
            {'role': 'user', 'content': user_message}
        ],
        stream=True # Включаем стриминг!
    )
    
    for chunk in stream:
        content = chunk['message']['content']
        if content:
            # Формируем SSE (Server-Sent Events) пакет
            yield f"data: {json.dumps({'text': content})}\n\n"
            
    yield "data: [DONE]\n\n"

@app.post("/chat/stream")
async def chat_stream(req: ChatRequest):
    return StreamingResponse(
        stream_ollama_response(req.message), 
        media_type="text/event-stream"
    )
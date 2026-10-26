from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

client = TestClient(app)

def test_read_root():
    response = client.get("/")
    assert response.status_code == 404 # У нас пока нет корневого эндпоинта, FastAPI вернет 404

# Тестируем стриминг (мокаем вызов Ollama)
@patch("main.stream_ollama_response")
def test_chat_stream(mock_stream):
    # Подменяем генератор на фейковый
    def fake_stream(msg):
        yield "data: {\"text\": \"Привет\"}\n\n"
        yield "data: [DONE]\n\n"
    mock_stream.return_value = fake_stream("test")

    response = client.post("/chat/stream", json={"message": "Привет"})
    assert response.status_code == 200
    assert "text/event-stream" in response.headers["content-type"]
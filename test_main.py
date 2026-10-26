from fastapi.testclient import TestClient
from main import app
from unittest.mock import patch

client = TestClient(app)

@patch("vector_db.add_document")
def test_add_knowledge(mock_add):
    response = client.post("/add_knowledge", json={"text": "Тестовый факт"})
    assert response.status_code == 200
    mock_add.assert_called_once_with("Тестовый факт")

@patch("main.stream_rag_response")
def test_chat_stream(mock_stream):
    def fake_stream(msg):
        yield "data: {\"text\": \"Привет\"}\n\n"
        yield "data: [DONE]\n\n"
    mock_stream.return_value = fake_stream("test")

    response = client.post("/chat/stream", json={"message": "Привет"})
    assert response.status_code == 200
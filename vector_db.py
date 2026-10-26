from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
import os

# Используем переменные окружения для гибкости
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
collection_name = "enterprise_kb"

def get_vector_store():
    """Ленивая инициализация Qdrant (только при вызове функции)."""
    client = QdrantClient(url=QDRANT_URL)
    try:
        client.get_collection(collection_name)
    except Exception:
        client.create_collection(
            collection_name=collection_name,
            vectors_config=models.VectorParams(size=768, distance=models.Distance.COSINE),
        )
    return QdrantVectorStore(client=client, collection_name=collection_name, embedding=embeddings)

def add_document(text: str):
    store = get_vector_store()
    doc = Document(page_content=text)
    store.add_documents([doc])

def search_docs(query: str, k: int = 2) -> str:
    store = get_vector_store()
    docs = store.similarity_search(query, k=k)
    if not docs:
        return "Контекст не найден."
    return "\n\n".join([d.page_content for d in docs])
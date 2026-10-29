from langchain_ollama import OllamaEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models
from langchain_core.documents import Document
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_community.document_compressors import FlashrankRerank
import os

OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")
QDRANT_URL = os.getenv("QDRANT_URL", "http://localhost:6333")

embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=OLLAMA_URL)
collection_name = "enterprise_kb"

def get_vector_store():
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

def search_docs(query: str, k: int = 3) -> str:
    store = get_vector_store()
    # 1. Ищем 6 документов на всякий случай
    base_retriever = store.as_retriever(search_kwargs={"k": 6})
    
    # 2. Оставляем 3 самых лучших 
    compressor = FlashrankRerank(model="ms-marco-MiniLM-L-12-v2", top_n=k)
    
    compression_retriever = ContextualCompressionRetriever(
        base_compressor=compressor, base_retriever=base_retriever
    )
    
    compressed_docs = compression_retriever.invoke(query)
    
    if not compressed_docs:
        return "Контекст не найден."
    
    return "\n\n".join([d.page_content for d in compressed_docs])
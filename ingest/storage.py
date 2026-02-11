import chromadb
from pathlib import Path
from .embedding import embed_text, embed_query
from .metadata import get_metadata
from .chunk_token import chunk_text
from .clean import clean_text

ROOT = Path(__file__).resolve().parent.parent

CHROMA_PATH = ROOT / "chroma_db"

def create_collection(client: chromadb.PersistentClient, name: str) -> chromadb.Collection:
    return client.get_or_create_collection(name)

def add_documents(collection: chromadb.Collection, chunks: list[str], embeddings: list[list[float]], metadatas: list[dict], ids: list[str]) -> None:
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)

def query_collection(collection: chromadb.Collection, query: str, n_results: int) -> list[dict]:
    return collection.query(query_embeddings=[embed_query(query)], n_results=n_results)

def delete_collection(client: chromadb.PersistentClient, name: str) -> None:
    client.delete_collection(name)

client = chromadb.PersistentClient(path=str(CHROMA_PATH))
collection = create_collection(client, "llm_concepts")
text = clean_text("llm_concepts_dirty.txt")
chunks = chunk_text(text)
metadatas, ids = get_metadata(text)
add_documents(collection, chunks, embed_text(text), metadatas, ids)
print(query_collection(collection, "What is RAG?", 1))
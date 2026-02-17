from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI
from .chunk_token import chunk_text
import os

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"

def embed_text(text: str) -> list[list[float]]:
    """Embed all chunks of text. Returns list of embedding vectors."""
    chunks = chunk_text(text)
    if not chunks:
        return []
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=chunks,
    )
    return [data.embedding for data in response.data]

def embed_query(query: str) -> list[float]:
    """Embed a single query string (e.g. for search). Same model as embed_text."""
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=query,
    )
    return response.data[0].embedding
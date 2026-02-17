"""Embeddings for indexing and querying; same model for both (text-embedding-3-small)."""

from pathlib import Path
import os

from dotenv import load_dotenv
from openai import OpenAI

from .chunk_token import chunk_text

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

EMBEDDING_MODEL = "text-embedding-3-small"

def embed_text(text: str) -> list[list[float]]:
    """Embed all chunks of text (using default chunk_text). Returns list of embedding vectors."""
    chunks = chunk_text(text)
    if not chunks:
        return []
    return embed_chunks(chunks)


def embed_chunks(chunks: list[str]) -> list[list[float]]:
    """Embed a list of strings. Use when chunks are produced by chunk_text_config (e.g. chroma config sweep)."""
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
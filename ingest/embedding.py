"""Embeddings for indexing and querying."""

from openai import OpenAI

from config import EMBEDDING_MODEL, OPENAI_API_KEY
from .chunk_token import chunk_text

client = OpenAI(api_key=OPENAI_API_KEY)

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
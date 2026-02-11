from .clean import clean_text
from pathlib import Path
from .metadata import get_metadata
from dotenv import load_dotenv
from openai import OpenAI
from .chunk_token import chunk_text
import os

DATA_DIR = Path(__file__).resolve().parent.parent / "data/llm_concepts_dirty.txt"

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

# print(embed_text(clean_text(DATA_DIR)))
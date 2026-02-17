"""ChromaDB client and collection access."""

import chromadb
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHROMA_PATH = ROOT / "chroma_db"


def get_client() -> chromadb.PersistentClient:
    return chromadb.PersistentClient(path=str(CHROMA_PATH))


def get_collection(name: str = "documents"):
    return get_client().get_or_create_collection(name)

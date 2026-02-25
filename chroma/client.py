"""ChromaDB client and collection access. Singleton client for graceful shutdown."""

from typing import Optional

import chromadb

from config import CHROMA_PERSIST_DIR, DEFAULT_COLLECTION, ROOT

CHROMA_PATH = CHROMA_PERSIST_DIR  # alias for __init__ exports
_client: Optional[chromadb.PersistentClient] = None


def get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=str(CHROMA_PERSIST_DIR))
    return _client


def close_client() -> None:
    """Release Chroma client (e.g. on graceful shutdown). Next get_client() will create a new one."""
    global _client
    _client = None


def get_collection(name: Optional[str] = None):
    name = name or DEFAULT_COLLECTION
    return get_client().get_or_create_collection(name)

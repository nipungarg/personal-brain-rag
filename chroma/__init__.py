"""ChromaDB storage and retrieval for RAG."""

from .client import ROOT, CHROMA_PATH, get_client, get_collection

__all__ = [
    "ROOT",
    "CHROMA_PATH",
    "get_client",
    "get_collection",
    "build_index",
    "add_documents",
    "get_relevant_chunks",
    "get_relevant_chunks_hybrid",
    "rerank_chunks",
    "generate_answer",
]

_STORE_ATTRS = {"build_index", "add_documents"}
_RETRIEVE_ATTRS = {"get_relevant_chunks", "get_relevant_chunks_hybrid", "rerank_chunks"}
_GENERATE_ATTRS = {"generate_answer"}


def __getattr__(name: str):
    if name in _STORE_ATTRS:
        from . import store
        return getattr(store, name)
    if name in _RETRIEVE_ATTRS:
        from . import retrieve
        return getattr(retrieve, name)
    if name in _GENERATE_ATTRS:
        from . import generate
        return getattr(generate, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

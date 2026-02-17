"""FAISS index and retrieval for RAG."""

__all__ = [
    "ROOT",
    "FAISS_DB",
    "FAISS_INDEX_PATH",
    "FAISS_METADATA_PATH",
    "load_index_and_metadata",
    "build_index_all",
    "save_index",
    "search_index",
    "retrieve_top_k",
    "generate_answer",
]

_CLIENT_ATTRS = {"ROOT", "FAISS_DB", "FAISS_INDEX_PATH", "FAISS_METADATA_PATH", "load_index_and_metadata"}
_STORE_ATTRS = {"build_index_all", "save_index", "search_index"}
_RETRIEVE_ATTRS = {"retrieve_top_k"}
_GENERATE_ATTRS = {"generate_answer"}


def __getattr__(name: str):
    if name in _CLIENT_ATTRS:
        from . import client
        return getattr(client, name)
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

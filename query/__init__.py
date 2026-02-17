"""In-memory RAG: load chunks, retrieve by similarity, prompt + generate."""

from .retrieve import load_all_chunks, retrieve_top_k_cross_corpus
from .prompt import build_prompt, format_context

__all__ = [
    "load_all_chunks",
    "retrieve_top_k_cross_corpus",
    "build_prompt",
    "format_context",
    "generate_response",
]


def __getattr__(name: str):
    if name == "generate_response":
        from .generate import generate_response
        return generate_response
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

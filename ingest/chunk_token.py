"""Token-based chunking for RAG: default splitter and configurable size/overlap for chroma sweep."""

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

_tokenizer = None


def _get_tokenizer():
    """Lazy-load tiktoken encoder (avoids slow import on server startup, e.g. Render)."""
    global _tokenizer
    if _tokenizer is None:
        _tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)
    return _tokenizer


def tik_token_len(text: str) -> int:
    """Token count for text (same model as embeddings)."""
    return len(_get_tokenizer().encode(text))


_text_splitter = None


def _get_text_splitter():
    global _text_splitter
    if _text_splitter is None:
        _text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=CHUNK_SIZE,
            chunk_overlap=CHUNK_OVERLAP,
            length_function=tik_token_len,
            separators=["\n\n", "\n", " ", ""],
        )
    return _text_splitter

def chunk_text(text: str) -> list[str]:
    """Split text with default size/overlap. Used by faiss_rag, query, embed_text."""
    return _get_text_splitter().split_text(text)


def chunk_text_config(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text with given chunk_size and chunk_overlap (token counts). For chroma config sweep."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tik_token_len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)
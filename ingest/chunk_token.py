"""Token-based chunking for RAG: default splitter and configurable size/overlap for chroma sweep."""

import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

from config import CHUNK_OVERLAP, CHUNK_SIZE, EMBEDDING_MODEL

tokenizer = tiktoken.encoding_for_model(EMBEDDING_MODEL)


def tik_token_len(text: str) -> int:
    """Token count for text (same model as embeddings)."""
    return len(tokenizer.encode(text))


text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=tik_token_len,
    separators=["\n\n", "\n", " ", ""],
)

def chunk_text(text: str) -> list[str]:
    """Split text with default size/overlap. Used by faiss_rag, query, embed_text."""
    return text_splitter.split_text(text)


def chunk_text_config(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    """Split text with given chunk_size and chunk_overlap (token counts). For chroma config sweep."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=tik_token_len,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_text(text)
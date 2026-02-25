from pathlib import Path

from ingest.chunk_token import chunk_text
from ingest.clean import clean_text
from ingest.embedding import embed_query, embed_text
from ingest.load import get_txt_filenames
from utils.similarity import cosine_similarity

SNIPPET_LEN = 200


def _preview(text: str, max_len: int = SNIPPET_LEN) -> str:
    """First max_len chars with '...' if truncated."""
    return text[:max_len] + "..." if len(text) > max_len else text


def load_all_chunks() -> list[dict]:
    """Load all .txt files from data/, clean, chunk, embed. Return list of dicts: embedding, text, source, chunk_id."""
    all_chunks = []
    for filename in get_txt_filenames():
        text = clean_text(filename)
        chunks = chunk_text(text)
        embeddings = embed_text(text)
        stem = Path(filename).stem
        for i, (emb, txt) in enumerate(zip(embeddings, chunks)):
            all_chunks.append({
                "embedding": emb,
                "text": txt,
                "source": filename,
                "chunk_id": f"{stem}_{i}",
            })
    return all_chunks


def retrieve_top_k_cross_corpus(
    query_embedding: list[float],
    all_chunks: list[dict],
    k: int,
) -> list[tuple[float, str, str]]:
    """Retrieve top-k across all documents. all_chunks from load_all_chunks(). Returns (score, chunk_id, preview)."""
    scored = [(cosine_similarity(query_embedding, c["embedding"]), c["chunk_id"], _preview(c["text"]))
              for c in all_chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]
from pathlib import Path
import numpy as np

from ingest.chunk_token import chunk_text
from ingest.clean import clean_text
from ingest.embedding import embed_query, embed_text
from ingest.load import get_txt_filenames


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


def cosine_similarity(a: list[float], b: list[float]) -> float:
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

def retrieve_top_k(
    query_embedding: list[float],
    chunk_embeddings: list[list[float]],
    chunk_texts: list[str],
    k: int,
    source_name: str = "doc",
) -> list[tuple[float, str, str]]:
    """Return top-k (score, chunk_id, preview) by cosine similarity. Preview = first 200 chars."""
    if len(chunk_embeddings) != len(chunk_texts):
        raise ValueError("chunk_embeddings and chunk_texts must have same length")
    scored = [
        (
            cosine_similarity(query_embedding, emb),
            f"{source_name}_{i}",
            (chunk_texts[i][:200] + "..." if len(chunk_texts[i]) > 200 else chunk_texts[i]),
        )
        for i, emb in enumerate(chunk_embeddings)
    ]
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]


def retrieve_top_k_cross_corpus(
    query_embedding: list[float],
    all_chunks: list[dict],
    k: int,
) -> list[tuple[float, str, str]]:
    """Retrieve top-k across all documents. all_chunks from load_all_chunks(). Returns (score, chunk_id, preview)."""
    scored = []
    for c in all_chunks:
        score = cosine_similarity(query_embedding, c["embedding"])
        preview = c["text"][:200] + "..." if len(c["text"]) > 200 else c["text"]
        scored.append((score, c["chunk_id"], preview))
    scored.sort(key=lambda x: x[0], reverse=True)
    return scored[:k]
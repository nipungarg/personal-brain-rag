import sys
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.chunk_token import chunk_text
from ingest.clean import clean_text
from ingest.embedding import embed_query, embed_text

text = clean_text("llm_concepts_dirty.txt")
chunks = chunk_text(text)
embeddings = embed_text(text)

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

results = retrieve_top_k(embed_query("What is RAG?"), embeddings, chunks, 3, source_name="llm_concepts_dirty")

# print("Top scores:", [r[0] for r in results])
# print("Chunk IDs:", [r[1] for r in results])
# print("First 200 characters of each chunk:")
# for score, chunk_id, preview in results:
#     print(f"  [{chunk_id}] {preview!r}")
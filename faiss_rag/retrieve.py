"""Retrieve top-k context from FAISS index."""

import faiss

from .client import load_index_and_metadata
from .store import search_index

# Max L2 distance for a result to be considered relevant.
# Above this, we return no context so the generator can say "I do not have enough information."
RELEVANCE_MAX_DISTANCE = 1.5


def retrieve_top_k(
    query_embedding: list[float],
    chunks: list[str],
    sources: list[str],
    index: faiss.Index,
    k: int = 3,
    max_distance: float = RELEVANCE_MAX_DISTANCE,
) -> list[tuple[str, str]]:
    """Return list of (chunk_text, source_filename) for top-k results within relevance threshold."""
    distances, indices = search_index(index, query_embedding, k)
    return [
        (chunks[int(idx)], sources[int(idx)])
        for idx, d in zip(indices, distances)
        if d <= max_distance
    ]

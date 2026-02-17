"""Retrieve context from ChromaDB."""

from ingest.embedding import embed_query
from .client import get_collection

# Max distance (L2 or cosine distance) for a result to be considered relevant.
# Above this, we return no context so the generator can say "I do not have enough information."
RELEVANCE_MAX_DISTANCE = 1.5


def get_relevant_chunks(
    query: str,
    n_results: int = 3,
    max_distance: float = RELEVANCE_MAX_DISTANCE,
    collection_name: str | None = None,
) -> list[dict]:
    """
    Return top-n chunks as list of dicts (source, chunk_index, id, text) within relevance threshold.
    collection_name: if None, use default "documents"; else use that collection (e.g. vault_small).
    """
    collection = get_collection(collection_name or "documents")
    results = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=n_results,
        include=["documents", "metadatas", "distances"],
    )
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0] or [{}] * len(ids)
    distances = results.get("distances", [[]])[0] or [0.0] * len(ids)

    chunks = []
    for i, (chunk_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
        if dist > max_distance:
            continue
        meta = meta if isinstance(meta, dict) else {}
        chunks.append({
            "source": meta.get("filename", ""),
            "chunk_index": meta.get("index", i),
            "id": chunk_id,
            "text": doc or "",
        })
    return chunks


def get_relevant_context(
    query: str,
    n_results: int = 3,
    max_distance: float = RELEVANCE_MAX_DISTANCE,
    collection_name: str | None = None,
) -> str:
    """Return concatenated top-n document chunks for the query, or empty string if no result is relevant."""
    chunks = get_relevant_chunks(
        query, n_results=n_results, max_distance=max_distance, collection_name=collection_name
    )
    return "\n---\n".join(c["text"] for c in chunks) if chunks else ""

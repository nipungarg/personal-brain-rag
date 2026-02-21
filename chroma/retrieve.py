"""Retrieve context from ChromaDB (dense and hybrid dense+sparse)."""

import re
from ingest.embedding import embed_query
from .client import get_collection

# Max distance (L2 or cosine distance) for a result to be considered relevant.
# Above this, we return no context so the generator can say "I do not have enough information."
RELEVANCE_MAX_DISTANCE = 1.5

# RRF constant for hybrid (Reciprocal Rank Fusion); higher k dampens rank differences.
RRF_K = 60

# Cross-encoder for reranking (lazy-loaded).
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L6-v2"
_reranker = None


def _get_cross_encoder():
    """Lazy-load the cross-encoder model so it's only loaded when reranking is used."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank_chunks(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    """
    Rerank chunks by cross-encoder (query, document) relevance. Improves top-k precision.

    chunks: list of dicts with "text" (and any other keys preserved in output).
    Returns same chunk dicts reordered by relevance score, length top_k.
    """
    if not chunks or top_k <= 0:
        return []
    if len(chunks) == 1:
        return chunks[:top_k]
    model = _get_cross_encoder()
    pairs = [(query, (c.get("text") or "")) for c in chunks]
    scores = model.predict(pairs)
    indexed = list(zip(scores, chunks))
    indexed.sort(key=lambda x: float(x[0]), reverse=True)
    return [c for _, c in indexed[:top_k]]


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25: lowercase, split on non-alphanumeric."""
    return re.sub(r"\W+", " ", (text or "").lower()).strip().split()


def _chunk_dict(meta: dict, chunk_id: str, doc: str, i: int) -> dict:
    """Build standard chunk dict (source, chunk_index, id, text) for retrieval results."""
    return {
        "source": meta.get("filename", ""),
        "chunk_index": meta.get("index", i),
        "id": chunk_id,
        "text": doc or "",
    }


def get_relevant_chunks(
    query: str,
    n_results: int = 3,
    max_distance: float = RELEVANCE_MAX_DISTANCE,
    collection_name: str | None = None,
    use_reranker: bool = False,
    rerank_initial_k: int = 20,
) -> list[dict]:
    """
    Return top-n chunks as list of dicts (source, chunk_index, id, text) within relevance threshold.
    collection_name: if None, use default "documents"; else use that collection (e.g. vault_small).
    use_reranker: if True, retrieve rerank_initial_k candidates then rerank with cross-encoder to n_results.
    rerank_initial_k: number of candidates to fetch when use_reranker (ignored otherwise).
    """
    collection = get_collection(collection_name or "documents")
    k = rerank_initial_k if use_reranker else n_results
    results = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0] or [{}] * len(ids)
    distances = results.get("distances", [[]])[0] or [0.0] * len(ids)

    chunks = []
    for i, (chunk_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
        if not use_reranker and dist > max_distance:
            continue
        meta = meta if isinstance(meta, dict) else {}
        chunks.append(_chunk_dict(meta, chunk_id, doc, i))
    if use_reranker:
        chunks = rerank_chunks(query, chunks, n_results)
    return chunks


def get_relevant_chunks_hybrid(
    query: str,
    n_results: int = 3,
    max_distance: float = RELEVANCE_MAX_DISTANCE,
    collection_name: str | None = None,
    top_k_dense: int = 20,
    top_k_sparse: int = 20,
    use_reranker: bool = False,
    rerank_initial_k: int = 20,
) -> list[dict]:
    """
    Hybrid retrieval: dense (Chroma) + sparse (BM25) with RRF.

    Fetches all documents from the collection to build BM25, runs dense query and
    BM25 over the query, then merges rankings with Reciprocal Rank Fusion. Only
    dense results within max_distance contribute to RRF so weak matches are not boosted.

    use_reranker: if True, take top rerank_initial_k by RRF then rerank with cross-encoder to n_results.
    Returns same format as get_relevant_chunks: list of dicts (source, chunk_index, id, text).
    """
    from rank_bm25 import BM25Okapi
    import numpy as np

    collection = get_collection(collection_name or "documents")
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    documents = all_data.get("documents") or []
    metadatas = all_data.get("metadatas") or [{}] * len(ids)
    if not documents:
        return []

    # Build sparse index (BM25)
    tokenized_corpus = [_tokenize(d or "") for d in documents]
    bm25 = BM25Okapi(tokenized_corpus)

    # Dense retrieval
    emb = embed_query(query)
    n_dense = min(top_k_dense, len(ids))
    dense_results = collection.query(
        query_embeddings=[emb],
        n_results=n_dense,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = dense_results["ids"][0]
    dense_distances = (dense_results.get("distances") or [[]])[0]
    if len(dense_distances) < len(dense_ids):
        dense_distances = [0.0] * len(dense_ids)

    # Sparse retrieval (BM25 top-k)
    tokenized_query = _tokenize(query)
    if not tokenized_query:
        sparse_scores = [0.0] * len(documents)
    else:
        sparse_scores = bm25.get_scores(tokenized_query)
    # Only include docs with positive BM25 score so we don't add non-matches to RRF.
    top_indices = np.argsort(sparse_scores)[::-1][:top_k_sparse]
    sparse_ids = [ids[i] for i in top_indices if sparse_scores[i] > 0]

    # RRF: only dense results within max_distance contribute
    rrf_scores = {}
    for rank, (chunk_id, dist) in enumerate(zip(dense_ids, dense_distances)):
        if dist <= max_distance:
            rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)
    for rank, chunk_id in enumerate(sparse_ids):
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank + 1)

    take_k = rerank_initial_k if use_reranker else n_results
    sorted_ids = sorted(rrf_scores.keys(), key=rrf_scores.get, reverse=True)[:take_k]
    id_to_doc = dict(zip(ids, documents))
    id_to_meta = {cid: (metadatas[i] if isinstance(metadatas[i], dict) else {}) for i, cid in enumerate(ids)}

    chunks = [_chunk_dict(id_to_meta.get(chunk_id, {}), chunk_id, id_to_doc.get(chunk_id) or "", i)
              for i, chunk_id in enumerate(sorted_ids)]
    if use_reranker:
        chunks = rerank_chunks(query, chunks, n_results)
    return chunks

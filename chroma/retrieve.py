"""Chroma retrieval: adaptive (heuristics + confidence fallback), dense, and hybrid (dense + BM25 RRF)."""

import re
import time
from ingest.embedding import embed_query
from .client import get_collection

RELEVANCE_MAX_DISTANCE = 1.5  # Max distance for a result to be considered relevant.
RRF_K = 60  # RRF constant for hybrid; higher k dampens rank differences.
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
    """Rerank chunks by cross-encoder (query, document) relevance; return top_k."""
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


def _is_keyword_heavy(query: str) -> bool:
    """Heuristic: many tokens or contains numbers → use hybrid retrieval."""
    if not query or not query.strip():
        return False
    return len(query.strip().split()) >= 4 or any(c.isdigit() for c in query)


def get_relevant_chunks_adaptive(
    query: str,
    n_results: int = 3,
    max_distance: float = RELEVANCE_MAX_DISTANCE,
    collection_name: str | None = None,
    top_k_dense: int = 20,
    top_k_sparse: int = 20,
    rerank_initial_k: int = 0,
    confidence_threshold: float = RELEVANCE_MAX_DISTANCE,
    return_timings: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """
    Heuristics + confidence fallback:
    Stage 1: If query looks keyword-heavy → use hybrid.
    Stage 2: Otherwise run dense; if dense top score (distance) > confidence_threshold → re-run hybrid.
    Else use dense result.
    rerank_initial_k: If > 0, retrieve that many and rerank to n_results; 0 = no reranking.
    If return_timings=True, returns (chunks, {"embed_s", "retrieval_s"}).
    """
    coll_name = collection_name or "documents"

    # Stage 1: keyword-heavy → hybrid
    if _is_keyword_heavy(query):
        result = get_relevant_chunks_hybrid(
            query,
            n_results=n_results,
            max_distance=max_distance,
            collection_name=coll_name,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            rerank_initial_k=rerank_initial_k,
            return_timings=return_timings,
        )
        return result

    # Stage 2: dense first, then confidence fallback
    use_reranker = rerank_initial_k > 0
    collection = get_collection(coll_name)
    if return_timings:
        t0 = time.perf_counter()
        emb = embed_query(query)
        embed_s = time.perf_counter() - t0
        t0 = time.perf_counter()
    else:
        emb = embed_query(query)

    k = rerank_initial_k if use_reranker else n_results
    results = collection.query(
        query_embeddings=[emb],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    ids = results["ids"][0]
    documents = results["documents"][0]
    metadatas = results.get("metadatas", [[]])[0] or [{}] * len(ids)
    distances = results.get("distances", [[]])[0] or [0.0] * len(ids)

    top_distance = float(distances[0]) if distances else float("inf")

    # Low confidence → fallback to hybrid
    if top_distance > confidence_threshold:
        if return_timings:
            retrieval_s = time.perf_counter() - t0
            result = get_relevant_chunks_hybrid(
                query,
                n_results=n_results,
                max_distance=max_distance,
                collection_name=coll_name,
                top_k_dense=top_k_dense,
                top_k_sparse=top_k_sparse,
                rerank_initial_k=rerank_initial_k,
                return_timings=True,
            )
            chunks, ht = result
            out_embed = embed_s
            out_retrieval = retrieval_s + ht["embed_s"] + ht["retrieval_s"]
            return chunks, {"embed_s": out_embed, "retrieval_s": out_retrieval}
        return get_relevant_chunks_hybrid(
            query,
            n_results=n_results,
            max_distance=max_distance,
            collection_name=coll_name,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            rerank_initial_k=rerank_initial_k,
            return_timings=False,
        )

    # Use dense result
    chunks = []
    for i, (chunk_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
        if not use_reranker and dist > max_distance:
            continue
        meta = meta if isinstance(meta, dict) else {}
        chunks.append(_chunk_dict(meta, chunk_id, doc, i))
    if use_reranker:
        chunks = rerank_chunks(query, chunks, n_results)

    if return_timings:
        retrieval_s = time.perf_counter() - t0
        return chunks, {"embed_s": embed_s, "retrieval_s": retrieval_s}
    return chunks


def get_relevant_chunks(
    query: str,
    n_results: int = 3,
    max_distance: float = RELEVANCE_MAX_DISTANCE,
    collection_name: str | None = None,
    rerank_initial_k: int = 0,
    return_timings: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """
    Return top-n chunks as list of dicts (source, chunk_index, id, text) within relevance threshold.
    rerank_initial_k: If > 0, retrieve that many and rerank to n_results; 0 = no reranking.
    If return_timings=True, returns (chunks, {"embed_s": float, "retrieval_s": float}).
    """
    use_reranker = rerank_initial_k > 0
    collection = get_collection(collection_name or "documents")
    k = rerank_initial_k if use_reranker else n_results

    if return_timings:
        t0 = time.perf_counter()
        emb = embed_query(query)
        embed_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        results = collection.query(
            query_embeddings=[emb],
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )
    else:
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

    if return_timings:
        retrieval_s = time.perf_counter() - t0
        return chunks, {"embed_s": embed_s, "retrieval_s": retrieval_s}
    return chunks


def get_relevant_chunks_hybrid(
    query: str,
    n_results: int = 3,
    max_distance: float = RELEVANCE_MAX_DISTANCE,
    collection_name: str | None = None,
    top_k_dense: int = 20,
    top_k_sparse: int = 20,
    rerank_initial_k: int = 0,
    return_timings: bool = False,
) -> list[dict] | tuple[list[dict], dict]:
    """
    Hybrid retrieval: dense (Chroma) + sparse (BM25) with RRF.
    rerank_initial_k: If > 0, take that many from RRF and rerank to n_results; 0 = no reranking.
    If return_timings=True, returns (chunks, {"embed_s": float, "retrieval_s": float}).
    """
    use_reranker = rerank_initial_k > 0
    from rank_bm25 import BM25Okapi
    import numpy as np

    collection = get_collection(collection_name or "documents")
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    documents = all_data.get("documents") or []
    metadatas = all_data.get("metadatas") or [{}] * len(ids)
    if not documents:
        return ([], {"embed_s": 0.0, "retrieval_s": 0.0}) if return_timings else []

    tokenized_corpus = [_tokenize(d or "") for d in documents]
    bm25 = BM25Okapi(tokenized_corpus)
    if return_timings:
        t0 = time.perf_counter()
        emb = embed_query(query)
        embed_s = time.perf_counter() - t0
        t0 = time.perf_counter()
    else:
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

    tokenized_query = _tokenize(query)
    sparse_scores = bm25.get_scores(tokenized_query) if tokenized_query else [0.0] * len(documents)
    top_indices = np.argsort(sparse_scores)[::-1][:top_k_sparse]
    sparse_ids = [ids[i] for i in top_indices if sparse_scores[i] > 0]
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

    if return_timings:
        retrieval_s = time.perf_counter() - t0
        return chunks, {"embed_s": embed_s, "retrieval_s": retrieval_s}
    return chunks

"""Chroma retrieval: adaptive (heuristics + confidence fallback), dense, and hybrid (dense + BM25 RRF)."""

import re
import time
from typing import List, Optional, Tuple, Union

from config import DEFAULT_COLLECTION, RELEVANCE_MAX_DISTANCE, RERANKER_MODEL, RRF_K
from ingest.embedding import embed_query
from utils.logging_config import get_logger
from .client import get_collection

_reranker = None
_log = get_logger(__name__)


def _log_retrieval_timings(embed_s: float, retrieval_s: float) -> None:
    _log.info(
        "retrieval_complete",
        extra={"embed_s": float(embed_s or 0), "retrieval_s": float(retrieval_s or 0)},
    )


def _get_cross_encoder():
    """Lazy-load cross-encoder for reranking."""
    global _reranker
    if _reranker is None:
        from sentence_transformers import CrossEncoder
        _reranker = CrossEncoder(RERANKER_MODEL)
    return _reranker


def rerank_chunks(
    query: str, chunks: list, top_k: int, return_scores: bool = False
) -> Union[List[dict], Tuple[List[dict], List[float]]]:
    """Rerank by cross-encoder; return top_k chunks, or (chunks, scores) if return_scores."""
    if not chunks or top_k <= 0:
        return ([], []) if return_scores else []
    if len(chunks) == 1:
        return (chunks[:top_k], [1.0]) if return_scores else chunks[:top_k]
    model = _get_cross_encoder()
    pairs = [(query, (c.get("text") or "")) for c in chunks]
    scores = model.predict(pairs)
    indexed = list(zip(scores, chunks))
    indexed.sort(key=lambda x: float(x[0]), reverse=True)
    top = indexed[:top_k]
    out_chunks = [c for _, c in top]
    if return_scores:
        return out_chunks, [float(s) for s, _ in top]
    return out_chunks


def _tokenize(text: str) -> list[str]:
    """Simple tokenizer for BM25: lowercase, split on non-alphanumeric."""
    return re.sub(r"\W+", " ", (text or "").lower()).strip().split()


def _chunk_dict(meta: dict, chunk_id: str, doc: str, i: int) -> dict:
    """Build chunk dict with source, chunk_index, id, text."""
    return {
        "source": meta.get("filename", ""),
        "chunk_index": meta.get("index", i),
        "id": chunk_id,
        "text": doc or "",
    }


def _is_keyword_heavy(query: str) -> bool:
    """Heuristic: many tokens or numbers → prefer hybrid."""
    if not query or not query.strip():
        return False
    return len(query.strip().split()) >= 4 or any(c.isdigit() for c in query)


def get_relevant_chunks_adaptive(
    query: str,
    n_results: int = 3,
    max_distance: float = RELEVANCE_MAX_DISTANCE,
    collection_name: Optional[str] = None,
    top_k_dense: int = 20,
    top_k_sparse: int = 20,
    rerank_initial_k: int = 0,
    confidence_threshold: float = RELEVANCE_MAX_DISTANCE,
    return_timings: bool = False,
) -> Union[List[dict], Tuple[List[dict], dict]]:
    """Keyword-heavy → hybrid; else dense with hybrid fallback if top score weak. Optional return_timings."""
    coll_name = collection_name or DEFAULT_COLLECTION

    if _is_keyword_heavy(query):
        return get_relevant_chunks_hybrid(
            query,
            n_results=n_results,
            max_distance=max_distance,
            collection_name=coll_name,
            top_k_dense=top_k_dense,
            top_k_sparse=top_k_sparse,
            rerank_initial_k=rerank_initial_k,
            return_timings=return_timings,
        )

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

    if top_distance > confidence_threshold:
        if return_timings:
            retrieval_s = time.perf_counter() - t0
            chunks, ht = get_relevant_chunks_hybrid(
                query,
                n_results=n_results,
                max_distance=max_distance,
                collection_name=coll_name,
                top_k_dense=top_k_dense,
                top_k_sparse=top_k_sparse,
                rerank_initial_k=rerank_initial_k,
                return_timings=True,
            )
            out_embed = embed_s
            out_retrieval = retrieval_s + ht["embed_s"] + ht["retrieval_s"]
            _log_retrieval_timings(out_embed, out_retrieval)
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

    chunks = []
    for i, (chunk_id, doc, meta, dist) in enumerate(zip(ids, documents, metadatas, distances)):
        if not use_reranker and dist > max_distance:
            continue
        meta = meta if isinstance(meta, dict) else {}
        chunks.append(_chunk_dict(meta, chunk_id, doc, i))
    if use_reranker:
        chunks, _ = rerank_chunks(query, chunks, n_results, return_scores=True)
    if return_timings:
        retrieval_s = time.perf_counter() - t0
        _log_retrieval_timings(embed_s, retrieval_s)
        return chunks, {"embed_s": embed_s, "retrieval_s": retrieval_s}
    return chunks


def get_relevant_chunks_hybrid(
    query: str,
    n_results: int = 3,
    max_distance: float = RELEVANCE_MAX_DISTANCE,
    collection_name: Optional[str] = None,
    top_k_dense: int = 20,
    top_k_sparse: int = 20,
    rerank_initial_k: int = 0,
    return_timings: bool = False,
) -> Union[List[dict], Tuple[List[dict], dict]]:
    """Dense + BM25 RRF. Optional rerank, return_timings."""
    import numpy as np
    from rank_bm25 import BM25Okapi

    use_reranker = rerank_initial_k > 0
    collection = get_collection(collection_name or DEFAULT_COLLECTION)
    all_data = collection.get(include=["documents", "metadatas"])
    ids = all_data["ids"]
    documents = all_data.get("documents") or []
    metadatas = all_data.get("metadatas") or [{}] * len(ids)
    if not documents:
        if return_timings:
            _log_retrieval_timings(0.0, 0.0)
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
        chunks, _ = rerank_chunks(query, chunks, n_results, return_scores=True)
    if return_timings:
        retrieval_s = time.perf_counter() - t0
        _log_retrieval_timings(embed_s, retrieval_s)
        return chunks, {"embed_s": embed_s, "retrieval_s": retrieval_s}
    return chunks

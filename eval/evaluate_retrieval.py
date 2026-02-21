"""Retrieval evaluation for chroma, query (in-memory), and faiss backends.

Only in-domain questions (expected_source set) are evaluated; out-of-domain are skipped.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import argparse
import statistics
import time

from ingest.embedding import embed_query
from utils.logger import log_event

from eval.common import load_questions, check_hit, snippet

TOP_K = 4
RELEVANCE_MAX_DISTANCE = 1.5  # Max distance for a result to count as "retrieved" (chroma/faiss).


def _result_row(rank: int, score: float, chunk_id: str, text_snippet: str, source: str) -> dict:
    """Single retrieval result dict for eval logging."""
    return {"rank": rank, "score": float(score), "chunk_id": chunk_id, "text_snippet": text_snippet, "source": source}


def _sources_from_results(results: list[dict]) -> list[str]:
    """Unique source filenames from results."""
    return list({r["source"] for r in results})


def _log_retrieval(backend: str, question: str, results: list[dict], hit: bool, elapsed: float) -> None:
    """Log one question's retrieval eval to run_log.jsonl."""
    log_event("retrieval_eval", {
        "backend": backend,
        "question": question,
        "results": results,
        "retrieved_sources": _sources_from_results(results),
        "hit": hit,
        "time_sec": round(elapsed, 4),
    })


def _run_retrieval_eval(
    backend: str,
    title: str,
    get_results_fn,
    apply_threshold_fn=None,
) -> None:
    """
    Generic retrieval eval loop: for each in-domain question, call get_results_fn(question)
    -> (results, elapsed), optionally apply threshold, compute hit, print and log.
    apply_threshold_fn: if set, applied to results before computing sources (chroma/faiss use distance threshold).
    """
    questions = load_questions()
    hits = total = 0
    elapsed_times: list[float] = []

    print(f"\n=== RETRIEVAL EVALUATION ({title}) ===\n")

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source")
        if expected_source is None:
            continue

        results, elapsed = get_results_fn(question)
        elapsed_times.append(elapsed)
        if apply_threshold_fn:
            results = apply_threshold_fn(results)
        retrieved_sources = _sources_from_results(results)
        hit = check_hit(retrieved_sources, expected_source)
        if hit:
            hits += 1
        total += 1

        print(f"Q: {question}")
        print(f"Retrieved sources: {retrieved_sources}")
        print(f"Hit: {hit}")
        print(f"Time: {elapsed:.3f}s\n")
        _log_retrieval(backend, question, results, hit, elapsed)

    recall = hits / total if total else 0.0
    avg_time = statistics.mean(elapsed_times) if elapsed_times else 0.0
    median_time = statistics.median(elapsed_times) if elapsed_times else 0.0
    print(f"=== RESULTS ({title}) ===")
    print(f"Recall@{TOP_K}: {recall:.2%} ({hits}/{total})")
    print(f"Time (avg): {avg_time:.3f}s  |  Time (median): {median_time:.3f}s")


def _chroma_get_results(collection_name: str):
    """Return (results, elapsed) getter for Chroma backend."""
    from chroma.client import get_collection

    collection = get_collection(collection_name or "documents")

    def get_results(question: str):
        start = time.time()
        raw = collection.query(
            query_embeddings=[embed_query(question)],
            n_results=TOP_K,
            include=["documents", "metadatas", "distances"],
        )
        elapsed = time.time() - start
        ids = raw["ids"][0]
        documents = raw["documents"][0]
        metadatas = raw["metadatas"][0]
        distances = raw.get("distances", [[]])[0] or [0.0] * len(ids)
        snip = lambda d: snippet(d) if d else ""
        results = [_result_row(r, score, chunk_id, snip(doc), meta.get("filename", ""))
                   for r, (chunk_id, doc, meta, score) in enumerate(zip(ids, documents, metadatas, distances), 1)]
        return results, elapsed

    return get_results


def _query_get_results():
    """Return (results, elapsed) getter for in-memory query backend."""
    from query.retrieve import load_all_chunks, retrieve_top_k_cross_corpus

    all_chunks = load_all_chunks()
    chunk_id_to_source = {c["chunk_id"]: c["source"] for c in all_chunks}

    def get_results(question: str):
        start = time.time()
        retrieved = retrieve_top_k_cross_corpus(embed_query(question), all_chunks, k=TOP_K)
        elapsed = time.time() - start
        results = [_result_row(rank, score, chunk_id, preview, chunk_id_to_source.get(chunk_id, ""))
                   for rank, (score, chunk_id, preview) in enumerate(retrieved, 1)]
        return results, elapsed

    return get_results


def _faiss_get_results():
    """Return (results, elapsed) getter for FAISS backend."""
    from faiss_rag.client import load_index_and_metadata
    from faiss_rag.store import search_index

    index, chunks, sources = load_index_and_metadata()

    def get_results(question: str):
        start = time.time()
        query_emb = embed_query(question)
        distances, indices = search_index(index, query_emb, k=TOP_K)
        elapsed = time.time() - start
        results = []
        for rank, (idx, dist) in enumerate(zip(indices, distances), 1):
            idx = int(idx)
            src = sources[idx]
            chunk_id = f"{Path(src).stem}_{sources[: idx + 1].count(src) - 1}"
            text = chunks[idx] if idx < len(chunks) else ""
            results.append(_result_row(rank, float(dist), chunk_id, snippet(text), src))
        return results, elapsed

    return get_results


def _apply_relevance_threshold(results: list[dict]) -> list[dict]:
    """Keep only results with score <= RELEVANCE_MAX_DISTANCE (chroma/faiss)."""
    return [r for r in results if r["score"] <= RELEVANCE_MAX_DISTANCE]


def evaluate_chroma(collection_name: str | None = None) -> None:
    """Evaluate retrieval for Chroma. collection_name: default 'documents'; use e.g. vault_small for sweep."""
    title = f"Chroma ({collection_name or 'documents'})"
    _run_retrieval_eval(
        "chroma",
        title,
        _chroma_get_results(collection_name),
        apply_threshold_fn=_apply_relevance_threshold,
    )


def evaluate_query() -> None:
    """Evaluate retrieval for in-memory query backend (cosine similarity)."""
    _run_retrieval_eval("query", "Query", _query_get_results())


def evaluate_faiss() -> None:
    """Evaluate retrieval for FAISS backend."""
    _run_retrieval_eval("faiss", "FAISS", _faiss_get_results(), apply_threshold_fn=_apply_relevance_threshold)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval for a backend.")
    parser.add_argument("backend", choices=["chroma", "query", "faiss"], help="Backend to evaluate.")
    parser.add_argument("--collection", default=None, help="Chroma collection name (for backend=chroma sweep).")
    args = parser.parse_args()

    if args.backend == "chroma":
        evaluate_chroma(collection_name=args.collection)
    elif args.backend == "query":
        evaluate_query()
    else:
        evaluate_faiss()


if __name__ == "__main__":
    main()

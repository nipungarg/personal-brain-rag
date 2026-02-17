"""Retrieval evaluation for chroma, query (in-memory), and faiss backends."""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from ingest.embedding import embed_query
from utils.logger import log_event


TOP_K = 4
# Only count results within this distance as "retrieved" (matches chroma/faiss retrieve threshold).
RELEVANCE_MAX_DISTANCE = 1.5


def load_questions() -> list[dict]:
    path = ROOT / "eval" / "questions.json"
    with open(path, "r") as f:
        return json.load(f)


def _check_hit(retrieved_sources: list[str], expected_source) -> bool:
    """
    True if retrieval matches expectation.
    - In-domain (expected_source set): at least one expected source in retrieved_sources.
    - Out-of-domain (expected_source None): hit iff retrieved_sources is empty (no relevant docs).
    """
    if expected_source is None:
        return not (retrieved_sources or [])
    if isinstance(expected_source, list):
        return any(exp in retrieved_sources for exp in expected_source)
    return expected_source in retrieved_sources


def _log_result(backend: str, question: str, results: list[dict], hit: bool, elapsed: float):
    """Log one question's evaluation with rank, score, chunk_id, text_snippet, source per result."""
    log_event("retrieval_eval", {
        "backend": backend,
        "question": question,
        "results": results,
        "retrieved_sources": list({r["source"] for r in results}),
        "hit": hit,
        "time_sec": round(elapsed, 4),
    })


def evaluate_chroma():
    """Evaluate retrieval using ChromaDB backend."""
    from chroma.client import get_collection

    collection = get_collection()
    questions = load_questions()
    total = hits = 0

    print("\n=== RETRIEVAL EVALUATION (Chroma) ===\n")

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source")

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
        distances = raw.get("distances", [[]])[0]
        if not distances and ids:
            distances = [0.0] * len(ids)

        results = []
        for r, (chunk_id, doc, meta, score) in enumerate(zip(ids, documents, metadatas, distances), 1):
            results.append({
                "rank": r,
                "score": float(score),
                "chunk_id": chunk_id,
                "text_snippet": (doc[:200] + "..." if len(doc) > 200 else doc) if doc else "",
                "source": meta.get("filename", ""),
            })
        # Apply same relevance threshold as chroma.retrieve: only results below max distance count.
        results = [r for r in results if r["score"] <= RELEVANCE_MAX_DISTANCE]

        retrieved_sources = list({r["source"] for r in results})
        hit = _check_hit(retrieved_sources, expected_source)
        if hit:
            hits += 1
        total += 1

        print(f"Q: {question}")
        print(f"Retrieved sources: {retrieved_sources}")
        print(f"Hit: {hit}")
        print(f"Time: {elapsed:.3f}s\n")

        _log_result("chroma", question, results, hit, elapsed)

    recall = hits / total if total else 0.0
    print("=== RESULTS (Chroma) ===")
    print(f"Recall@{TOP_K}: {recall:.2%} ({hits}/{total})")


def evaluate_query():
    """Evaluate retrieval using in-memory query backend (cosine similarity)."""
    from query.retrieve import load_all_chunks, retrieve_top_k_cross_corpus

    all_chunks = load_all_chunks()
    chunk_id_to_source = {c["chunk_id"]: c["source"] for c in all_chunks}
    questions = load_questions()
    total = hits = 0

    print("\n=== RETRIEVAL EVALUATION (Query) ===\n")

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source")

        start = time.time()
        retrieved = retrieve_top_k_cross_corpus(embed_query(question), all_chunks, k=TOP_K)
        elapsed = time.time() - start

        results = []
        for r, (score, chunk_id, preview) in enumerate(retrieved, 1):
            results.append({
                "rank": r,
                "score": float(score),
                "chunk_id": chunk_id,
                "text_snippet": preview,
                "source": chunk_id_to_source.get(chunk_id, ""),
            })

        retrieved_sources = list({r["source"] for r in results})
        hit = _check_hit(retrieved_sources, expected_source)
        if hit:
            hits += 1
        total += 1

        print(f"Q: {question}")
        print(f"Retrieved sources: {retrieved_sources}")
        print(f"Hit: {hit}")
        print(f"Time: {elapsed:.3f}s\n")

        _log_result("query", question, results, hit, elapsed)

    recall = hits / total if total else 0.0
    print("=== RESULTS (Query) ===")
    print(f"Recall@{TOP_K}: {recall:.2%} ({hits}/{total})")


def evaluate_faiss():
    """Evaluate retrieval using FAISS backend."""
    from faiss_rag.client import load_index_and_metadata
    from faiss_rag.store import search_index

    index, chunks, sources = load_index_and_metadata()
    questions = load_questions()
    total = hits = 0

    print("\n=== RETRIEVAL EVALUATION (FAISS) ===\n")

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source")

        start = time.time()
        query_emb = embed_query(question)
        distances, indices = search_index(index, query_emb, k=TOP_K)
        elapsed = time.time() - start

        results = []
        for r, (idx, dist) in enumerate(zip(indices, distances), 1):
            idx = int(idx)
            src = sources[idx]
            local_i = sources[: idx + 1].count(src) - 1
            chunk_id = f"{Path(src).stem}_{local_i}"
            text = chunks[idx] if idx < len(chunks) else ""
            text_snippet = text[:200] + "..." if len(text) > 200 else text
            results.append({
                "rank": r,
                "score": float(dist),
                "chunk_id": chunk_id,
                "text_snippet": text_snippet,
                "source": src,
            })
        # Apply same relevance threshold as faiss_rag.retrieve.
        results = [r for r in results if r["score"] <= RELEVANCE_MAX_DISTANCE]

        retrieved_sources = list({r["source"] for r in results})
        hit = _check_hit(retrieved_sources, expected_source)
        if hit:
            hits += 1
        total += 1

        print(f"Q: {question}")
        print(f"Retrieved sources: {retrieved_sources}")
        print(f"Hit: {hit}")
        print(f"Time: {elapsed:.3f}s\n")

        _log_result("faiss", question, results, hit, elapsed)

    recall = hits / total if total else 0.0
    print("=== RESULTS (FAISS) ===")
    print(f"Recall@{TOP_K}: {recall:.2%} ({hits}/{total})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval for a backend.")
    parser.add_argument(
        "backend",
        choices=["chroma", "query", "faiss"],
        help="Backend to evaluate: chroma, query, or faiss",
    )
    args = parser.parse_args()

    if args.backend == "chroma":
        evaluate_chroma()
    elif args.backend == "query":
        evaluate_query()
    else:
        evaluate_faiss()


if __name__ == "__main__":
    main()

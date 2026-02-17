"""Generation evaluation for chroma, query, and faiss backends.

Runs each backend's generate_answer(), checks that returned sources match expected_source
(in-domain: at least one expected source cited; out-of-domain: no sources cited).
Logs question, retrieved_sources, hit, time, and answer snippet.
"""

import argparse
import statistics
import time

from utils.logger import log_event

from .common import load_questions, check_hit

TOP_K = 4
EVAL_TEMPERATURE = 0.0  # Deterministic recall for comparable runs across collections.


def _log_generation(
    backend: str,
    question: str,
    retrieved_sources: list[str],
    hit: bool,
    elapsed: float,
    answer_snippet: str = "",
) -> None:
    """Log one question's generation eval to run_log.jsonl."""
    snippet = answer_snippet[:200] + "..." if len(answer_snippet) > 200 else answer_snippet
    log_event("generation_eval", {
        "backend": backend,
        "question": question,
        "retrieved_sources": retrieved_sources,
        "hit": hit,
        "time_sec": round(elapsed, 4),
        "answer_snippet": snippet,
    })


def _run_generation_eval(backend: str, title: str, generate_fn) -> None:
    """
    Generic generation eval loop: for each question, call generate_fn(question)
    -> {answer, sources}, compute hit from sources vs expected_source, print and log.
    """
    questions = load_questions()
    hits = total = 0
    elapsed_times: list[float] = []

    print(f"\n=== GENERATION EVALUATION ({title}) ===\n")

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source")

        start = time.time()
        out = generate_fn(question)
        elapsed = time.time() - start
        elapsed_times.append(elapsed)

        returned_sources = out.get("sources", [])
        answer = out.get("answer", "")
        hit = check_hit(returned_sources, expected_source)
        if hit:
            hits += 1
        total += 1

        print(f"Q: {question}")
        print(f"Retrieved sources (from LLM): {returned_sources}")
        print(f"Hit: {hit}")
        print(f"Time: {elapsed:.3f}s\n")
        _log_generation(backend, question, returned_sources, hit, elapsed, answer)

    recall = hits / total if total else 0.0
    avg_time = statistics.mean(elapsed_times) if elapsed_times else 0.0
    median_time = statistics.median(elapsed_times) if elapsed_times else 0.0
    print(f"=== RESULTS ({title}) ===")
    print(f"Recall: {recall:.2%} ({hits}/{total})")
    print(f"Time (avg): {avg_time:.3f}s  |  Time (median): {median_time:.3f}s")


def evaluate_chroma(collection_name: str | None = None) -> None:
    """Evaluate generation for Chroma. collection_name: default 'documents'; use e.g. vault_small for sweep."""
    from chroma.generate import generate_answer

    title = f"Chroma: {collection_name or 'documents'}"

    def generate_fn(q):
        return generate_answer(q, n_results=TOP_K, collection_name=collection_name, temperature=EVAL_TEMPERATURE)

    _run_generation_eval("chroma", title, generate_fn)


def evaluate_query() -> None:
    """Evaluate generation for in-memory query backend."""
    from query.generate import generate_answer

    def generate_fn(q):
        return generate_answer(q, n_results=TOP_K)

    _run_generation_eval("query", "Query", generate_fn)


def evaluate_faiss() -> None:
    """Evaluate generation for FAISS backend."""
    from faiss_rag.generate import generate_answer

    def generate_fn(q):
        return generate_answer(q, n_results=TOP_K)

    _run_generation_eval("faiss", "FAISS", generate_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generation (LLM answer + sources) for a backend.")
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

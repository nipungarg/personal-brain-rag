"""Generation evaluation for chroma, query, and faiss backends."""

import argparse
import statistics
import time

from utils.logger import log_event

from config import DEFAULT_COLLECTION, EVAL_TEMPERATURE, EVAL_TOP_K
from eval.common import load_questions, check_hit, snippet  # side effect: project root on path


def _log_generation(
    backend: str,
    question: str,
    retrieved_sources: list[str],
    hit: bool,
    elapsed: float,
    answer_snippet: str = "",
) -> None:
    """Log one question's generation eval to run_log.jsonl."""
    log_event("generation_eval", {
        "backend": backend,
        "question": question,
        "retrieved_sources": retrieved_sources,
        "hit": hit,
        "time_sec": round(elapsed, 4),
        "answer_snippet": snippet(answer_snippet),
    })


def _run_generation_eval(backend: str, title: str, generate_fn) -> None:
    """
    Generic generation eval loop: for each question, call generate_fn(question)
    -> {answer, sources, ...}; compute hit, accumulate tokens and cost; print and log.
    """
    questions = load_questions()
    hits = total = 0
    elapsed_times: list[float] = []
    total_tokens = 0
    total_cost_usd = 0.0

    print(f"\n=== GENERATION EVALUATION ({title}) ===\n")

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source")

        start = time.time()
        out = generate_fn(question)
        elapsed = time.time() - start
        elapsed_times.append(elapsed)
        total_tokens += out.get("total_tokens", 0)
        total_cost_usd += out.get("cost_usd", 0.0)

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
    print(f"Total tokens: {total_tokens}")
    print(f"Cost: ${total_cost_usd:.4f}")


def evaluate_chroma(
    collection_name: str | None = None,
    rerank_initial_k: int = 0,
) -> None:
    """Evaluate generation for Chroma (adaptive retrieval). collection_name: default 'documents'; use e.g. vault_small for sweep."""
    from chroma.generate import generate_answer

    parts = [collection_name or DEFAULT_COLLECTION]
    if rerank_initial_k > 0:
        parts.append(f"rerank({rerank_initial_k})")
    title = f"Chroma: {'+'.join(parts)}"

    def generate_fn(q):
        return generate_answer(
            q,
            n_results=EVAL_TOP_K,
            collection_name=collection_name,
            temperature=EVAL_TEMPERATURE,
            rerank_initial_k=rerank_initial_k,
        )

    _run_generation_eval("chroma", title, generate_fn)


def evaluate_query() -> None:
    """Evaluate generation for in-memory query backend."""
    from query.generate import generate_answer

    def generate_fn(q):
        return generate_answer(q, n_results=EVAL_TOP_K)

    _run_generation_eval("query", "Query", generate_fn)


def evaluate_faiss() -> None:
    """Evaluate generation for FAISS backend."""
    from faiss_rag.generate import generate_answer

    def generate_fn(q):
        return generate_answer(q, n_results=EVAL_TOP_K)

    _run_generation_eval("faiss", "FAISS", generate_fn)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generation (LLM answer + sources) for a backend.")
    parser.add_argument("backend", choices=["chroma", "query", "faiss"], help="Backend to evaluate.")
    parser.add_argument("--collection", default=None, help="Chroma collection name (for backend=chroma sweep).")
    parser.add_argument("--rerank", type=int, default=0, dest="rerank_initial_k", metavar="K", help="If > 0, retrieve K candidates and rerank (chroma). 0 = no reranking.")
    args = parser.parse_args()

    if args.backend == "chroma":
        evaluate_chroma(
            collection_name=args.collection,
            rerank_initial_k=args.rerank_initial_k,
        )
    elif args.backend == "query":
        evaluate_query()
    else:
        evaluate_faiss()


if __name__ == "__main__":
    main()

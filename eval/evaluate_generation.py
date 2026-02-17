"""Generation evaluation for chroma, query (in-memory), and faiss backends.

Runs each backend's generate_answer(), checks that returned sources match
expected_source (in-domain: at least one expected source cited; out-of-domain:
no sources cited). Logs question, retrieved_sources, hit, time, and answer snippet.
"""

import argparse
import json
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

from utils.logger import log_event


TOP_K = 4


def load_questions() -> list[dict]:
    """Load eval questions from eval/questions.json."""
    with open(ROOT / "eval" / "questions.json", "r") as f:
        return json.load(f)


def _check_hit(returned_sources: list[str], expected_source) -> bool:
    """
    True if generation output matches expectation.
    - In-domain (expected_source set): at least one expected source in returned_sources.
    - Out-of-domain (expected_source None): hit iff returned_sources is empty.
    """
    if expected_source is None:
        return not (returned_sources or [])
    if isinstance(expected_source, list):
        return any(exp in returned_sources for exp in expected_source)
    return expected_source in returned_sources


def _log_result(
    backend: str,
    question: str,
    retrieved_sources: list[str],
    hit: bool,
    elapsed: float,
    answer_snippet: str = "",
) -> None:
    """Log one question's generation eval."""
    log_event("generation_eval", {
        "backend": backend,
        "question": question,
        "retrieved_sources": retrieved_sources,
        "hit": hit,
        "time_sec": round(elapsed, 4),
        "answer_snippet": answer_snippet[:200] + "..." if len(answer_snippet) > 200 else answer_snippet,
    })


def evaluate_chroma() -> None:
    """Evaluate generation using ChromaDB backend."""
    from chroma.generate import generate_answer

    questions = load_questions()
    total = hits = 0

    print("\n=== GENERATION EVALUATION (Chroma) ===\n")

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source")

        start = time.time()
        out = generate_answer(question, n_results=TOP_K)
        elapsed = time.time() - start

        returned_sources = out.get("sources", [])
        answer = out.get("answer", "")
        hit = _check_hit(returned_sources, expected_source)

        if hit:
            hits += 1
        total += 1

        print(f"Q: {question}")
        print(f"Retrieved sources (from LLM): {returned_sources}")
        print(f"Hit: {hit}")
        print(f"Time: {elapsed:.3f}s\n")

        _log_result("chroma", question, returned_sources, hit, elapsed, answer)

    recall = hits / total if total else 0.0
    print("=== RESULTS (Chroma) ===")
    print(f"Recall: {recall:.2%} ({hits}/{total})")


def evaluate_query() -> None:
    """Evaluate generation using in-memory query backend."""
    from query.generate import generate_answer

    questions = load_questions()
    total = hits = 0

    print("\n=== GENERATION EVALUATION (Query) ===\n")

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source")

        start = time.time()
        out = generate_answer(question, n_results=TOP_K)
        elapsed = time.time() - start

        returned_sources = out.get("sources", [])
        answer = out.get("answer", "")
        hit = _check_hit(returned_sources, expected_source)

        if hit:
            hits += 1
        total += 1

        print(f"Q: {question}")
        print(f"Retrieved sources (from LLM): {returned_sources}")
        print(f"Hit: {hit}")
        print(f"Time: {elapsed:.3f}s\n")

        _log_result("query", question, returned_sources, hit, elapsed, answer)

    recall = hits / total if total else 0.0
    print("=== RESULTS (Query) ===")
    print(f"Recall: {recall:.2%} ({hits}/{total})")


def evaluate_faiss() -> None:
    """Evaluate generation using FAISS backend."""
    from faiss_rag.generate import generate_answer

    questions = load_questions()
    total = hits = 0

    print("\n=== GENERATION EVALUATION (FAISS) ===\n")

    for item in questions:
        question = item["question"]
        expected_source = item.get("expected_source")

        start = time.time()
        out = generate_answer(question, n_results=TOP_K)
        elapsed = time.time() - start

        returned_sources = out.get("sources", [])
        answer = out.get("answer", "")
        hit = _check_hit(returned_sources, expected_source)

        if hit:
            hits += 1
        total += 1

        print(f"Q: {question}")
        print(f"Retrieved sources (from LLM): {returned_sources}")
        print(f"Hit: {hit}")
        print(f"Time: {elapsed:.3f}s\n")

        _log_result("faiss", question, returned_sources, hit, elapsed, answer)

    recall = hits / total if total else 0.0
    print("=== RESULTS (FAISS) ===")
    print(f"Recall: {recall:.2%} ({hits}/{total})")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate generation (LLM response + sources) for a backend.")
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

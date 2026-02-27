#!/usr/bin/env python3
"""Ask a question via user input; run RAG and print answer, sources, timings, tokens, cost.

Usage:
  python ask.py
  python ask.py --question "What is RAG?"
  python ask.py -q "How does tokenization work?"
  python ask.py -q "Fresh run, no cache" --no-cache
"""

import sys
import time
import uuid

from config import DEFAULT_BACKEND, DEFAULT_COLLECTION, DEFAULT_TOP_K, ensure_root_path
ensure_root_path()

from utils.logging_config import configure_logging, get_logger, set_request_id
configure_logging()

import argparse

_log = get_logger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ask a question (user input or --question); run RAG and print answer."
    )
    parser.add_argument(
        "-q", "--question",
        default=None,
        help="Question to ask. If omitted, prompt for input.",
    )
    parser.add_argument(
        "--backend",
        choices=["chroma", "query", "faiss"],
        default=DEFAULT_BACKEND,
        help="RAG backend.",
    )
    parser.add_argument("--collection", default=None, help="Chroma collection (backend=chroma).")
    parser.add_argument("--rerank", type=int, default=0, dest="rerank_initial_k", metavar="K", help="If > 0, retrieve K candidates and rerank (chroma). 0 = no reranking.")
    parser.add_argument("--no-cache", action="store_false", dest="use_cache", default=True, help="Disable semantic response cache (chroma).")
    parser.add_argument("-n", "--n-results", type=int, default=DEFAULT_TOP_K, help="Number of chunks to retrieve.")
    args = parser.parse_args()

    question = args.question
    if question is None or question.strip() == "":
        question = input("Question: ").strip()
    if not question:
        print("No question provided.", file=sys.stderr)
        sys.exit(1)

    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    _log.info(
        "request_start",
        extra={"question_preview": question[:80], "backend": args.backend},
    )
    t0 = time.perf_counter()
    try:
        if args.backend == "chroma":
            from chroma.generate import generate_answer
            out = generate_answer(
                question,
                n_results=args.n_results,
                collection_name=args.collection or DEFAULT_COLLECTION,
                rerank_initial_k=args.rerank_initial_k,
                use_cache=args.use_cache,
                return_timings=True,
            )
        elif args.backend == "query":
            from query.generate import generate_answer
            out = generate_answer(question, n_results=args.n_results)
            out.setdefault("cached", False)
        else:
            from faiss_rag.generate import generate_answer
            out = generate_answer(question, n_results=args.n_results)
            out.setdefault("cached", False)
    except Exception as e:
        _log.exception("request_error", extra={"error": str(e)})
        set_request_id(None)
        raise
    total_s = time.perf_counter() - t0
    if "total_s" not in out:
        out["total_s"] = total_s
    _log.info(
        "request_complete",
        extra={
            "total_s": out["total_s"],
            "cached": out.get("cached", False),
            "embed_s": out.get("embed_s"),
            "retrieval_s": out.get("retrieval_s"),
            "llm_s": out.get("llm_s"),
            "cost_usd": out.get("cost_usd"),
        },
    )

    # --- Metadata ---
    print("\n---")
    print("Cached:", out.get("cached", False))
    print("Total time: {:.3f}s".format(out.get("total_s", total_s)))
    if "embed_s" in out:
        print("  Embedding: {:.3f}s".format(out["embed_s"]))
    if "retrieval_s" in out:
        print("  Retrieval / vector search: {:.3f}s".format(out["retrieval_s"]))
    if "llm_s" in out:
        print("  LLM generation: {:.3f}s".format(out["llm_s"]))
    print("Total tokens:", out.get("total_tokens", 0))
    print("Cost: ${:.4f}".format(out.get("cost_usd", 0.0)))
    print("---\n")
    print("ANSWER:\n", out.get("answer", ""))
    print("\nSOURCES:", out.get("sources", []))


if __name__ == "__main__":
    main()

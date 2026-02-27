"""Generate RAG answers using ChromaDB-retrieved context."""

import time
import uuid
from typing import Optional

from config import CACHE_SIM_THRESHOLD, DEFAULT_TOP_K, TEMPERATURE
from query.prompt import build_prompt
from query.llm import complete_rag
from query.cache import get_cached_response, set_cached_response
from utils.logging_config import get_logger, get_request_id, set_request_id

from .retrieve import get_relevant_chunks_adaptive

_log = get_logger(__name__)


def generate_answer(
    query: str,
    n_results: int = DEFAULT_TOP_K,
    collection_name: Optional[str] = None,
    temperature: float = TEMPERATURE,
    rerank_initial_k: int = 0,
    use_cache: bool = False,
    cache_threshold: float = CACHE_SIM_THRESHOLD,
    return_timings: bool = False,
) -> dict:
    """Adaptive retrieval + optional cache/rerank. Returns answer, sources, cached; optional timings/tokens/cost."""
    if get_request_id() is None:
        set_request_id(str(uuid.uuid4())[:8])
    try:
        return _generate_answer_impl(
            query=query,
            n_results=n_results,
            collection_name=collection_name,
            temperature=temperature,
            rerank_initial_k=rerank_initial_k,
            use_cache=use_cache,
            cache_threshold=cache_threshold,
            return_timings=return_timings,
        )
    except Exception as e:
        _log.exception("generate_error", extra={"error": str(e)})
        raise


def _generate_answer_impl(
    query: str,
    n_results: int = DEFAULT_TOP_K,
    collection_name: Optional[str] = None,
    temperature: float = TEMPERATURE,
    rerank_initial_k: int = 0,
    use_cache: bool = False,
    cache_threshold: float = CACHE_SIM_THRESHOLD,
    return_timings: bool = False,
) -> dict:
    def _raw_generate(q: str, with_timings: bool = False) -> dict:
        result = get_relevant_chunks_adaptive(
            q,
            n_results=n_results,
            collection_name=collection_name,
            rerank_initial_k=rerank_initial_k,
            return_timings=with_timings,
        )
        if with_timings:
            chunks, timings = result
            embed_s, retrieval_s = timings["embed_s"], timings["retrieval_s"]
        else:
            chunks = result
        prompt = build_prompt(q, chunks)
        out = complete_rag(prompt, temperature=temperature)
        if with_timings:
            llm_s = out.get("llm_s", 0.0)
            out["embed_s"] = embed_s
            out["retrieval_s"] = retrieval_s
            out["llm_s"] = llm_s
            out["total_s"] = embed_s + retrieval_s + llm_s
        return out

    if use_cache:
        t0 = time.perf_counter()
        cached = get_cached_response(query, threshold=cache_threshold)
        if cached is not None:
            total_s = time.perf_counter() - t0
            _log.info("cache_hit", extra={"total_s": total_s})
            return {
                **cached,
                "cached": True,
                "embed_s": 0.0,
                "retrieval_s": total_s,
                "llm_s": 0.0,
                "total_s": total_s,
            }
        out = _raw_generate(query, with_timings=return_timings)
        set_cached_response(query, out.get("answer", ""), out.get("sources", []))
        out["cached"] = False
        return out

    out = _raw_generate(query, with_timings=return_timings)
    out.setdefault("cached", False)
    return out


if __name__ == "__main__":
    q = "How is prompt engineering different from tokenization?"
    out = generate_answer(q)
    print("ANSWER:", out["answer"])
    print("SOURCES:", out["sources"])

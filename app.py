#!/usr/bin/env python3
"""Gradio UI: question input, streaming answer, rerank toggle."""

import time
import uuid

from config import ensure_root_path
ensure_root_path()

from utils.logging_config import configure_logging, get_logger, set_request_id
configure_logging()

import gradio as gr

from config import CACHE_SIM_THRESHOLD, DEFAULT_TOP_K, RERANK_K, TEMPERATURE
from chroma.retrieve import get_relevant_chunks_adaptive
from query.prompt import build_prompt
from query.llm import complete_rag_stream
from query.cache import get_cached_response, set_cached_response

_log = get_logger(__name__)


def run_rag(question: str, use_rerank: bool):
    """Retrieve, build prompt, stream LLM; yield (answer, sources, time, cost). Uses semantic cache on hit."""
    if not (question or "").strip():
        yield "", "", "", ""
        return
    request_id = str(uuid.uuid4())[:8]
    set_request_id(request_id)
    _log.info(
        "request_start",
        extra={"question_preview": (question.strip() or "")[:80], "use_rerank": use_rerank},
    )
    yield "Searching…", "…", "…", "…"
    t0 = time.perf_counter()
    try:
        cached = get_cached_response(question.strip(), threshold=CACHE_SIM_THRESHOLD)
        if cached is not None:
            total_s = time.perf_counter() - t0
            _log.info("cache_hit", extra={"total_s": total_s})
            answer = cached.get("answer", "")
            sources = cached.get("sources", [])
            sources_str = ", ".join(sources) if sources else "—"
            yield answer, sources_str, f"{total_s:.2f} s", "$0.0000"
            return
        rerank_k = RERANK_K if use_rerank else 0
        chunks, timings = get_relevant_chunks_adaptive(
            question.strip(),
            n_results=DEFAULT_TOP_K,
            rerank_initial_k=rerank_k,
            return_timings=True,
        )
        _log.info(
            "retrieval_done",
            extra={"embed_s": timings["embed_s"], "retrieval_s": timings["retrieval_s"]},
        )
        prompt = build_prompt(question.strip(), chunks)
        yield "Generating…", "…", "…", "…"
        accumulated = ""
        for delta, final in complete_rag_stream(prompt, temperature=TEMPERATURE):
            if delta:
                accumulated += delta
                total_s = time.perf_counter() - t0
                yield accumulated, "…", f"{total_s:.2f} s", "…"
            elif final is not None:
                total_s = time.perf_counter() - t0
                _log.info(
                    "request_complete",
                    extra={
                        "total_s": total_s,
                        "llm_s": final.get("llm_s"),
                        "prompt_tokens": final.get("prompt_tokens", 0),
                        "completion_tokens": final.get("completion_tokens", 0),
                        "cost_usd": final.get("cost_usd", 0.0),
                    },
                )
                answer = final.get("answer", accumulated)
                sources = final.get("sources", [])
                sources_str = ", ".join(sources) if sources else "—"
                cost_str = f"${final.get('cost_usd', 0.0):.4f}"
                set_cached_response(question.strip(), answer, sources)
                yield answer, sources_str, f"{total_s:.2f} s", cost_str
    except Exception as e:
        _log.exception("request_error", extra={"error": str(e)})
        set_request_id(None)
        raise


with gr.Blocks(title="Personal Brain RAG") as demo:
    gr.Markdown("# 🧠 Personal Brain (RAG Vault)")
    gr.Markdown("Ask questions about your indexed documents. Toggle **Use rerank** to rerank retrieval with a cross-encoder.")
    with gr.Row():
        question = gr.Textbox(
            label="Question",
            placeholder="e.g. What is RAG?",
            lines=2,
            scale=4,
        )
        use_rerank = gr.Checkbox(label="Use rerank", value=False, scale=1)
    submit_btn = gr.Button("Submit", variant="primary")
    answer_out = gr.Textbox(label="Answer", lines=12, interactive=False)
    sources_out = gr.Textbox(label="Sources", lines=1, interactive=False)
    with gr.Row():
        time_out = gr.Textbox(label="Total time", interactive=False, scale=1)
        cost_out = gr.Textbox(label="Cost", interactive=False, scale=1)
    submit_btn.click(
        fn=run_rag,
        inputs=[question, use_rerank],
        outputs=[answer_out, sources_out, time_out, cost_out],
        show_progress="full",
    )


if __name__ == "__main__":
    demo.launch()

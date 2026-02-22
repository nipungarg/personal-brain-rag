"""Shared LLM call for RAG generation: single place for model, max_tokens, and prompt → parsed answer + sources."""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .prompt import parse_response

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = "gpt-4o-mini"
MAX_TOKENS = 1000

# Pricing per 1M tokens (update from https://platform.openai.com/docs/pricing if needed)
INPUT_PRICE_PER_1M = 0.15
OUTPUT_PRICE_PER_1M = 0.60

_client = None


def get_client() -> OpenAI:
    """Lazy singleton OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def complete_rag(prompt: str, temperature: float = 0.2) -> dict:
    """
    Send prompt to the LLM and return parsed {answer, sources} plus usage, cost, and llm_s.

    Returns:
        answer, sources, prompt_tokens, completion_tokens, total_tokens, cost_usd, llm_s
    """
    t0 = time.perf_counter()
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    parsed = parse_response(response.choices[0].message.content)
    parsed["llm_s"] = time.perf_counter() - t0
    usage = getattr(response, "usage", None)
    if usage is not None:
        pt, ct = getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0
        total = getattr(usage, "total_tokens", None) or (pt + ct)
        cost_usd = (pt / 1e6 * INPUT_PRICE_PER_1M) + (ct / 1e6 * OUTPUT_PRICE_PER_1M)
        parsed["prompt_tokens"] = pt
        parsed["completion_tokens"] = ct
        parsed["total_tokens"] = total
        parsed["cost_usd"] = cost_usd
    else:
        parsed.setdefault("prompt_tokens", 0)
        parsed.setdefault("completion_tokens", 0)
        parsed.setdefault("total_tokens", 0)
        parsed.setdefault("cost_usd", 0.0)
    return parsed

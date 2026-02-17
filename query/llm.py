"""Shared LLM call for RAG generation: single place for model, max_tokens, and prompt → parsed answer + sources."""

import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

from .prompt import parse_response

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

MODEL = "gpt-4o-mini"
MAX_TOKENS = 1000

_client = None


def get_client() -> OpenAI:
    """Lazy singleton OpenAI client."""
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def complete_rag(prompt: str, temperature: float = 0.2) -> dict:
    """
    Send prompt to the LLM and return parsed {answer, sources}.
    Used by chroma, faiss_rag, and query generate modules.
    """
    response = get_client().chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=MAX_TOKENS,
    )
    return parse_response(response.choices[0].message.content)

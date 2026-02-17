"""Generate RAG answers using ChromaDB-retrieved context."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from query.prompt import build_prompt, parse_response

from .client import ROOT
from .retrieve import get_relevant_chunks

load_dotenv(ROOT / ".env")
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def generate_answer(query: str, n_results: int = 3) -> dict:
    """
    Retrieve context from ChromaDB and generate an answer.
    Returns {"answer": str, "sources": list[str]} (sources = filenames used).
    """
    chunks = get_relevant_chunks(query, n_results=n_results)
    prompt = build_prompt(query, chunks)
    response = _get_client().chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    return parse_response(response.choices[0].message.content)


if __name__ == "__main__":
    q = "How is prompt engineering different from tokenization?"
    out = generate_answer(q)
    print("ANSWER:", out["answer"])
    print("SOURCES:", out["sources"])

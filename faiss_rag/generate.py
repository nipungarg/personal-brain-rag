"""Generate RAG answers using FAISS-retrieved context."""

import os

from dotenv import load_dotenv
from openai import OpenAI

from ingest.embedding import embed_query
from query.prompt import build_prompt, parse_response

from .client import ROOT, load_index_and_metadata
from .retrieve import retrieve_top_k

load_dotenv(ROOT / ".env")
_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


def _retrieved_to_context(retrieved: list[tuple[str, str]]) -> list[dict]:
    """Convert (chunk_text, source) to dicts expected by build_prompt."""
    return [
        {
            "source": source,
            "chunk_index": i,
            "id": f"{source}_{i}",
            "text": text,
        }
        for i, (text, source) in enumerate(retrieved)
    ]


def generate_answer(query: str, n_results: int = 3) -> dict:
    """
    Retrieve context from FAISS index and generate an answer.
    Returns {"answer": str, "sources": list[str]} (sources = filenames used).
    """
    index, chunks, sources = load_index_and_metadata()
    query_embedding = embed_query(query)
    retrieved = retrieve_top_k(query_embedding, chunks, sources, index, k=n_results)
    context = _retrieved_to_context(retrieved)
    prompt = build_prompt(query, context)
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

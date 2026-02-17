import os
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

from ingest.embedding import embed_query
from .prompt import build_prompt, parse_response
from .retrieve import load_all_chunks, retrieve_top_k_cross_corpus


def _retrieve_results_to_context(retrieved: list[tuple[float, str, str]]) -> list[dict]:
    """Convert retrieve_top_k output (score, chunk_id, preview) to dicts expected by build_prompt."""
    context = []
    for score, chunk_id, preview in retrieved:
        if "_" in chunk_id and chunk_id.split("_")[-1].isdigit():
            parts = chunk_id.rsplit("_", 1)
            source, idx = parts[0], int(parts[1])
        else:
            source, idx = chunk_id, 0
        context.append({
            "source": source,
            "chunk_index": idx,
            "id": chunk_id,
            "text": preview,
        })
    return context


def generate_response(query: str, context: list[tuple[float, str, str]]) -> dict:
    """
    Generate answer from pre-retrieved context.
    Returns {"answer": str, "sources": list[str]}.
    """
    prompt = build_prompt(query, _retrieve_results_to_context(context))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    return parse_response(response.choices[0].message.content)


# Min cosine similarity to consider a chunk relevant (below = treat as no context for out-of-domain).
MIN_SIMILARITY = 0.5


def generate_answer(query: str, n_results: int = 3, min_similarity: float = MIN_SIMILARITY) -> dict:
    """
    Retrieve from in-memory chunks (with relevance threshold) and generate an answer.
    Returns {"answer": str, "sources": list[str]} (sources = filenames used).
    """
    all_chunks = load_all_chunks()
    retrieved = retrieve_top_k_cross_corpus(embed_query(query), all_chunks, n_results)
    filtered = [(s, cid, p) for s, cid, p in retrieved if s >= min_similarity]
    return generate_response(query, filtered)


if __name__ == "__main__":
    query = "How prompt engineering is different from tokenization?"
    all_chunks = load_all_chunks()
    retrieved = retrieve_top_k_cross_corpus(embed_query(query), all_chunks, 3)
    print("Top Retrieved Chunks:")
    for i, (score, chunk_id, _) in enumerate(retrieved, 1):
        print(f"{i}. ID: {chunk_id} | Score: {score:.2f}")
    out = generate_response(query, retrieved)
    print("\nANSWER:", out["answer"])
    print("SOURCES:", out["sources"])
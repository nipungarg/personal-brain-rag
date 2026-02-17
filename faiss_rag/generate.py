"""Generate RAG answers using FAISS-retrieved context."""

from ingest.embedding import embed_query
from query.prompt import build_prompt
from query.llm import complete_rag

from .client import load_index_and_metadata
from .retrieve import retrieve_top_k


def _retrieved_to_context(retrieved: list[tuple[str, str]]) -> list[dict]:
    """Convert (chunk_text, source) to dicts expected by build_prompt."""
    return [
        {"source": source, "chunk_index": i, "id": f"{source}_{i}", "text": text}
        for i, (text, source) in enumerate(retrieved)
    ]


def generate_answer(query: str, n_results: int = 3) -> dict:
    """
    Retrieve context from FAISS index and generate an answer.

    Returns:
        {"answer": str, "sources": list[str]} — answer text and cited filenames.
    """
    index, chunks, sources = load_index_and_metadata()
    query_embedding = embed_query(query)
    retrieved = retrieve_top_k(query_embedding, chunks, sources, index, k=n_results)
    context = _retrieved_to_context(retrieved)
    prompt = build_prompt(query, context)
    return complete_rag(prompt, temperature=0.2)


if __name__ == "__main__":
    q = "How is prompt engineering different from tokenization?"
    out = generate_answer(q)
    print("ANSWER:", out["answer"])
    print("SOURCES:", out["sources"])

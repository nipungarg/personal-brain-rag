"""Generate RAG answers using ChromaDB-retrieved context."""

from query.prompt import build_prompt
from query.llm import complete_rag

from .retrieve import get_relevant_chunks, get_relevant_chunks_hybrid


def generate_answer(
    query: str,
    n_results: int = 3,
    collection_name: str | None = None,
    temperature: float = 0.2,
    use_hybrid: bool = False,
    use_reranker: bool = False,
    rerank_initial_k: int = 20,
) -> dict:
    """
    Retrieve context from ChromaDB and generate an answer.

    Args:
        query: User question.
        n_results: Number of chunks to retrieve (top-k).
        collection_name: Chroma collection; None => "documents". Use e.g. vault_small for sweep.
        temperature: 0 for deterministic eval; 0.2 for slightly varied answers.
        use_hybrid: If True, use hybrid (dense + BM25) retrieval with RRF instead of dense-only.
        use_reranker: If True, rerank candidates with a cross-encoder before returning top n_results.
        rerank_initial_k: Number of candidates to retrieve when use_reranker (ignored otherwise).

    Returns:
        {"answer": str, "sources": list[str]} — answer text and cited filenames.
    """
    get_chunks = get_relevant_chunks_hybrid if use_hybrid else get_relevant_chunks
    chunks = get_chunks(
        query,
        n_results=n_results,
        collection_name=collection_name,
        use_reranker=use_reranker,
        rerank_initial_k=rerank_initial_k,
    )
    prompt = build_prompt(query, chunks)
    return complete_rag(prompt, temperature=temperature)


if __name__ == "__main__":
    q = "How is prompt engineering different from tokenization?"
    out = generate_answer(q)
    print("ANSWER:", out["answer"])
    print("SOURCES:", out["sources"])

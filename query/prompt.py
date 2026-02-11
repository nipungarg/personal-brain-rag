from typing import List, Dict


def format_context(chunks: List[Dict]) -> str:
    """
    Format retrieved chunks into a readable context block.
    Each chunk includes metadata for traceability.
    """

    formatted_chunks = []

    for chunk in chunks:
        chunk_block = (
            f"[Source: {chunk['source']} | "
            f"Chunk: {chunk['chunk_index']} | "
            f"ID: {chunk['id']}]\n"
            f"{chunk['text']}\n"
        )
        formatted_chunks.append(chunk_block)

    return "\n---\n".join(formatted_chunks)


def build_prompt(question: str, retrieved_chunks: List[Dict]) -> str:
    """
    Build a grounded RAG prompt using retrieved context.
    """

    context_block = format_context(retrieved_chunks)

    prompt = f"""
You are a precise and reliable assistant.

You must answer the question using ONLY the provided context below.
If the answer is not explicitly contained in the context, respond with:
"I cannot find the answer in the provided documents."

Do NOT use outside knowledge.
Do NOT guess.
Be concise and factual.

CONTEXT:
========
{context_block}
========

QUESTION:
{question}

ANSWER:
"""

    return prompt.strip()

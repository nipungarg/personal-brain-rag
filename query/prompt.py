"""RAG prompt building and response parsing: format context, build_prompt, parse_response (answer + SOURCES line)."""


def format_context(chunks: list[dict]) -> str:
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


def build_prompt(question: str, retrieved_chunks: list[dict]) -> str:
    """
    Build a grounded RAG prompt using retrieved context.
    Asks the model to list source filenames at the end in a parseable format.
    """
    context_block = format_context(retrieved_chunks)

    prompt = f"""
You are a precise and reliable assistant.

You must answer the question using ONLY the provided context below.
If the answer is not explicitly contained in the context, respond with:
"I do not have enough information."

Do NOT use outside knowledge.
Do NOT guess.
Be concise and factual.

At the end of your answer, on a new line, list the source filenames you used in this exact format:
SOURCES: filename1.txt, filename2.txt
If you did not use any context, write:
SOURCES: none

CONTEXT:
========
{context_block}
========

QUESTION:
{question}

ANSWER:
"""
    return prompt.strip()


def parse_response(response: str) -> dict[str, str | list[str]]:
    """
    Parse model response into answer text and list of source filenames.
    Expects a final line "SOURCES: ..." or "SOURCES: none".
    Returns {"answer": str, "sources": list[str]}.
    """
    response = (response or "").strip()
    if not response:
        return {"answer": "", "sources": []}

    lines = response.split("\n")
    answer_lines = []
    sources = []

    for line in lines:
        if line.strip().upper().startswith("SOURCES:"):
            rest = line.split(":", 1)[-1].strip()
            if rest.lower() != "none":
                sources = [s.strip() for s in rest.split(",") if s.strip()]
            break
        answer_lines.append(line)

    answer = "\n".join(answer_lines).strip()
    return {"answer": answer, "sources": sources}

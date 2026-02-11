from retrieve import retrieve_top_k
from prompt import build_prompt
from openai import OpenAI
from dotenv import load_dotenv
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ingest.embedding import embed_query
from ingest.embedding import embed_text
from ingest.chunk_token import chunk_text
from ingest.clean import clean_text

text = clean_text("llm_concepts_dirty.txt")
chunks = chunk_text(text)
embeddings = embed_text(text)
query = "How prompt engineering is different from tokenization?"

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def _retrieve_results_to_context(retrieved: list[tuple[float, str, str]]) -> list[dict]:
    """Convert retrieve_top_k output (score, chunk_id, preview) to dicts expected by build_prompt."""
    context = []
    for score, chunk_id, preview in retrieved:
        # chunk_id is like "llm_concepts_dirty_0" -> source="llm_concepts_dirty", chunk_index=0
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


def generate_response(query: str, context: list[tuple[float, str, str]]) -> str:
    prompt = build_prompt(query, _retrieve_results_to_context(context))
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=1000,
    )
    return response.choices[0].message.content

retrieved = retrieve_top_k(embed_query(query), embeddings, chunks, 3, source_name="llm_concepts_dirty")

print("Top Retrieved Chunks:")
for i, (score, chunk_id, preview) in enumerate(retrieved, start=1):
    print(f"{i}. ID: {chunk_id} | Score: {score:.2f}")
print()

answer = generate_response(query, retrieved)
print("ANSWER:")
print(answer)
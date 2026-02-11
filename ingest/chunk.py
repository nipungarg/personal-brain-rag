from .clean import clean_text
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data/llm_concepts_dirty.txt"

def chunk_text(text: str, size=500, overlap=100) -> list[str]:
    chunks = []
    for i in range(0, len(text), size-overlap):
        chunks.append(text[i:i+size])
    return chunks

print((chunk_text(clean_text(DATA_DIR)))[0])
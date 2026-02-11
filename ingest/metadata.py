from .clean import clean_text
from pathlib import Path
from .chunk_token import chunk_text
import uuid

DATA_DIR = Path(__file__).resolve().parent.parent / "data/llm_concepts_dirty.txt"

def get_metadata(text: str) -> tuple[list[dict], list[str]]:
    chunks = chunk_text(text)
    metadatas = []
    ids = []
    for i, chunk in enumerate(chunks):
        metadatas.append({
            "index": i,
            "id": str(uuid.uuid4()),
            "filename": DATA_DIR.name,
        })
        ids.append(f'{DATA_DIR.name}_{i}')
    return metadatas, ids

# print(get_metadata(clean_text(DATA_DIR))[0][1])
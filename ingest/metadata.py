from pathlib import Path
from .chunk_token import chunk_text
import uuid


def get_metadata(text: str, source_filename: str) -> tuple[list[dict], list[str]]:
    """Return (metadatas, ids) for each chunk. source_filename is the data file name (e.g. llm_concepts_dirty.txt)."""
    chunks = chunk_text(text)
    metadatas = []
    ids = []
    stem = Path(source_filename).stem
    for i, chunk in enumerate(chunks):
        metadatas.append({
            "index": i,
            "id": str(uuid.uuid4()),
            "filename": source_filename,
        })
        ids.append(f"{stem}_{i}")
    return metadatas, ids
from pathlib import Path
from .chunk_token import chunk_text
import uuid


def get_metadata(text: str, source_filename: str) -> tuple[list[dict], list[str]]:
    """Return (metadatas, ids) for each chunk. source_filename is the data file name. Uses default chunk_text."""
    chunks = chunk_text(text)
    return get_metadata_for_chunks(chunks, source_filename)


def get_metadata_for_chunks(chunks: list[str], source_filename: str) -> tuple[list[dict], list[str]]:
    """Return (metadatas, ids) for pre-computed chunks. Used when chunking with custom size/overlap (e.g. chroma sweep)."""
    metadatas = []
    ids = []
    stem = Path(source_filename).stem
    for i in range(len(chunks)):
        metadatas.append({
            "index": i,
            "id": str(uuid.uuid4()),
            "filename": source_filename,
        })
        ids.append(f"{stem}_{i}")
    return metadatas, ids
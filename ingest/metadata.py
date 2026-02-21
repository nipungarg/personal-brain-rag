import uuid
from pathlib import Path


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
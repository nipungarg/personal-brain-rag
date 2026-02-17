"""FAISS index paths and load index + metadata from disk."""

import pickle
from pathlib import Path

import faiss

ROOT = Path(__file__).resolve().parent.parent
FAISS_DB = ROOT / "faiss_db"
FAISS_INDEX_PATH = FAISS_DB / "faiss.index"
FAISS_METADATA_PATH = FAISS_DB / "faiss_metadata.pkl"


def load_index_and_metadata() -> tuple[faiss.Index, list[str], list[str]]:
    """Load persisted index and (chunks, sources) from disk."""
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta["chunks"], meta["sources"]

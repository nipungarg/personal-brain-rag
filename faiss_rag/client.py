"""FAISS index paths and load index + metadata from disk."""

import pickle
import faiss

from config import FAISS_DB, FAISS_INDEX_PATH, FAISS_METADATA_PATH, ROOT


def load_index_and_metadata() -> tuple[faiss.Index, list[str], list[str]]:
    """Load persisted index and (chunks, sources) from disk."""
    index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, "rb") as f:
        meta = pickle.load(f)
    return index, meta["chunks"], meta["sources"]

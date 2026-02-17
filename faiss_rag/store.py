"""Build and save FAISS index from data/ .txt files."""

import pickle
import faiss
import numpy as np

from ingest.embedding import embed_text
from ingest.clean import clean_text
from ingest.chunk_token import chunk_text
from ingest.load import get_txt_filenames

from .client import FAISS_DB, FAISS_INDEX_PATH, FAISS_METADATA_PATH


def build_index_all() -> tuple[faiss.Index, list[str], list[str]]:
    """Load all .txt from data/, clean, chunk, embed; build one index. Return (index, chunks, sources)."""
    all_embeddings = []
    all_chunks = []
    all_sources = []
    for filename in get_txt_filenames():
        text = clean_text(filename)
        chunks = chunk_text(text)
        embeddings = embed_text(text)
        for emb, txt in zip(embeddings, chunks):
            all_embeddings.append(emb)
            all_chunks.append(txt)
            all_sources.append(filename)
    embeddings = np.array(all_embeddings).astype("float32")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, all_chunks, all_sources


def save_index(index: faiss.Index, chunks: list[str], sources: list[str]) -> None:
    """Persist index and metadata to disk."""
    FAISS_DB.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    with open(FAISS_METADATA_PATH, "wb") as f:
        pickle.dump({"chunks": chunks, "sources": sources}, f)


def search_index(index: faiss.Index, query_embedding: list[float], k: int = 3) -> tuple[np.ndarray, np.ndarray]:
    query_vector = np.array([query_embedding]).astype("float32")
    distances, indices = index.search(query_vector, k)
    return distances[0], indices[0]


if __name__ == "__main__":
    index, chunks, sources = build_index_all()
    save_index(index, chunks, sources)
    print(f"Built index with {len(chunks)} chunks from {len(set(sources))} files.")

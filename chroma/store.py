"""Build ChromaDB index from data/ .txt files.

- build_index(): default collection "documents" with default chunk size/overlap.
- build_index_for_config(): named collection (e.g. vault_small) with custom chunk_size/chunk_overlap for sweep.
"""

import chromadb

from ingest.embedding import embed_chunks
from ingest.metadata import get_metadata_for_chunks
from ingest.chunk_token import chunk_text, chunk_text_config
from ingest.clean import clean_text
from ingest.load import get_txt_filenames

from .client import get_client


def create_collection(client: chromadb.PersistentClient, name: str) -> chromadb.Collection:
    return client.get_or_create_collection(name)


def add_documents(
    collection: chromadb.Collection,
    chunks: list[str],
    embeddings: list[list[float]],
    metadatas: list[dict],
    ids: list[str],
) -> None:
    collection.add(documents=chunks, embeddings=embeddings, metadatas=metadatas, ids=ids)


def build_index(collection_name: str = "documents") -> chromadb.Collection:
    """Load all .txt from data/, clean, chunk (default size/overlap), embed; add to ChromaDB. Return collection."""
    client = get_client()
    collection = create_collection(client, collection_name)
    for filename in get_txt_filenames():
        text = clean_text(filename)
        chunks = chunk_text(text)
        if not chunks:
            continue
        metadatas, ids = get_metadata_for_chunks(chunks, filename)
        add_documents(collection, chunks, embed_chunks(chunks), metadatas, ids)
    return collection


def build_index_for_config(
    collection_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> chromadb.Collection:
    """Build a Chroma collection with given chunk_size and chunk_overlap (tokens). Recreates collection if it exists."""
    client = get_client()
    try:
        client.delete_collection(collection_name)
    except Exception:
        pass
    collection = create_collection(client, collection_name)
    for filename in get_txt_filenames():
        text = clean_text(filename)
        chunks = chunk_text_config(text, chunk_size, chunk_overlap)
        if not chunks:
            continue
        embeddings = embed_chunks(chunks)
        metadatas, ids = get_metadata_for_chunks(chunks, filename)
        add_documents(collection, chunks, embeddings, metadatas, ids)
    return collection


if __name__ == "__main__":
    build_index()

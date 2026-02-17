"""Build and query ChromaDB index from data/ .txt files."""

import chromadb
from pathlib import Path

from ingest.embedding import embed_text, embed_query
from ingest.metadata import get_metadata
from ingest.chunk_token import chunk_text
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


def query_collection(collection: chromadb.Collection, query: str, n_results: int = 3) -> list[dict]:
    return collection.query(query_embeddings=[embed_query(query)], n_results=n_results)


def delete_collection(client: chromadb.PersistentClient, name: str) -> None:
    client.delete_collection(name)


def build_index(collection_name: str = "documents") -> chromadb.Collection:
    """Load all .txt from data/, clean, chunk, embed; add to ChromaDB. Return collection."""
    client = get_client()
    collection = create_collection(client, collection_name)
    for filename in get_txt_filenames():
        text = clean_text(filename)
        chunks = chunk_text(text)
        metadatas, ids = get_metadata(text, filename)
        add_documents(collection, chunks, embed_text(text), metadatas, ids)
    return collection


if __name__ == "__main__":
    collection = build_index()
    # print(query_collection(collection, "What is RAG?", 1))

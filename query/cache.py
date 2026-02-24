"""Semantic response cache: SQLite-backed, match by query embedding similarity."""

import json
import sqlite3
from pathlib import Path

from ingest.embedding import embed_query

ROOT = Path(__file__).resolve().parent.parent
CACHE_DB = ROOT / "cache.db"

# Minimum cosine similarity (0–1) to return a cached response.
DEFAULT_THRESHOLD = 0.90


def _cosine_similarity(a: list[float], b: list[float]) -> float:
    """Cosine similarity between two vectors."""
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(x * x for x in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def _init_db(conn: sqlite3.Connection) -> None:
    conn.execute("""
        CREATE TABLE IF NOT EXISTS cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            query_embedding TEXT NOT NULL,
            query_text TEXT,
            answer TEXT NOT NULL,
            sources TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()


def get_cached_response(query: str, threshold: float = DEFAULT_THRESHOLD) -> dict | None:
    """Return cached {answer, sources} if query embedding similarity >= threshold, else None."""
    conn = sqlite3.connect(CACHE_DB)
    _init_db(conn)
    try:
        query_emb = embed_query(query)
        rows = conn.execute(
            "SELECT query_embedding, answer, sources FROM cache"
        ).fetchall()
    finally:
        conn.close()

    if not rows:
        return None

    best_sim = -1.0
    best_answer = best_sources = None
    for (emb_json, answer, sources_json) in rows:
        cached_emb = json.loads(emb_json)
        sim = _cosine_similarity(query_emb, cached_emb)
        if sim > best_sim:
            best_sim = sim
            best_answer = answer
            best_sources = json.loads(sources_json) if sources_json else []

    if best_sim >= threshold:
        return {
            "answer": best_answer,
            "sources": best_sources,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost_usd": 0.0,
        }
    return None


def set_cached_response(query: str, answer: str, sources: list) -> None:
    """Store a response in the cache (query is embedded and stored with answer/sources)."""
    query_emb = embed_query(query)
    conn = sqlite3.connect(CACHE_DB)
    _init_db(conn)
    try:
        conn.execute(
            "INSERT INTO cache (query_embedding, query_text, answer, sources) VALUES (?, ?, ?, ?)",
            (json.dumps(query_emb), query[:2000], answer, json.dumps(sources)),
        )
        conn.commit()
    finally:
        conn.close()

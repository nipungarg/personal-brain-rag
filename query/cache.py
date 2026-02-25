"""Semantic response cache: SQLite-backed, match by query embedding similarity."""

import json
import sqlite3
from typing import Optional

from config import CACHE_DB, CACHE_SIM_THRESHOLD
from ingest.embedding import embed_query
from utils.similarity import cosine_similarity


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


def get_cached_response(query: str, threshold: float = CACHE_SIM_THRESHOLD) -> Optional[dict]:
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
        sim = cosine_similarity(query_emb, cached_emb)
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

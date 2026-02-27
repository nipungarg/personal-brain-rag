"""Config interface: loads .env and exposes all settings. .env is the source; this module is the interface."""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Project root (directory containing config.py)
ROOT = Path(__file__).resolve().parent
load_dotenv(ROOT / ".env")


def ensure_root_path() -> None:
    """Ensure project root is on sys.path (for entry points: app, ask, run_server, eval)."""
    root_str = str(ROOT)
    if root_str not in sys.path:
        sys.path.insert(0, root_str)

# --- Secrets (from .env only; do not commit .env) ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")

# --- Models ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
LLM_MODEL = os.getenv("LLM_MODEL", "gpt-4o-mini")

# --- Retrieval ---
DEFAULT_TOP_K = int(os.getenv("DEFAULT_TOP_K", "4"))
RELEVANCE_MAX_DISTANCE = float(os.getenv("RELEVANCE_MAX_DISTANCE", "1.5"))  # max L2/cosine distance for relevance
RRF_K = int(os.getenv("RRF_K", "60"))  # RRF constant for hybrid
RERANKER_MODEL = os.getenv("RERANKER_MODEL", "cross-encoder/ms-marco-MiniLM-L6-v2")
RERANK_K = int(os.getenv("RERANK_K", "20"))  # candidates to retrieve when rerank enabled (app / ask)
CACHE_SIM_THRESHOLD = float(os.getenv("CACHE_SIM_THRESHOLD", "0.90"))  # min cosine similarity for cache hit
MIN_SIMILARITY = float(os.getenv("MIN_SIMILARITY", "0.5"))  # min cosine similarity for query backend

# --- Backend ---
DEFAULT_BACKEND = os.getenv("DEFAULT_BACKEND", "chroma")
DEFAULT_COLLECTION = os.getenv("DEFAULT_COLLECTION", "documents")

# --- Paths (vault/cache live only here; app is stateless) ---
DATA_DIR = Path(os.getenv("DATA_DIR", str(ROOT / "data")))
CHROMA_PERSIST_DIR = Path(os.getenv("CHROMA_PERSIST_DIR", str(ROOT / "chroma_db")))
FAISS_DB = Path(os.getenv("FAISS_DB", str(ROOT / "faiss_db")))
FAISS_INDEX_PATH = FAISS_DB / "faiss.index"
FAISS_METADATA_PATH = FAISS_DB / "faiss_metadata.pkl"
CACHE_DB = ROOT / "cache.db"
QUESTIONS_PATH = ROOT / "eval" / "questions.json"
LOG_FILE = os.getenv("LOG_FILE", "eval/run_log.jsonl")

# --- Chunking (default splitter) ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))

# --- Runtime ---
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "1000"))
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.2"))
INPUT_PRICE_PER_1M = float(os.getenv("INPUT_PRICE_PER_1M", "0.15"))
OUTPUT_PRICE_PER_1M = float(os.getenv("OUTPUT_PRICE_PER_1M", "0.60"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")

# --- Eval ---
EVAL_TOP_K = int(os.getenv("EVAL_TOP_K", "4"))
EVAL_TEMPERATURE = float(os.getenv("EVAL_TEMPERATURE", "0.0"))

# --- Server (long-lived service) ---
SERVER_HOST = os.getenv("SERVER_HOST", "0.0.0.0")
# Use PORT when set (e.g. Render, Heroku); else SERVER_PORT or 7860
SERVER_PORT = int(os.getenv("PORT") or os.getenv("SERVER_PORT", "7860"))

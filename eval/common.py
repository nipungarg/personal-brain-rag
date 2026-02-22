"""Shared eval utilities: question loading, hit logic, and snippet for retrieval/generation eval."""

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
QUESTIONS_PATH = ROOT / "eval" / "questions.json"


def snippet(text: str, max_len: int = 200) -> str:
    """Truncate to max_len chars with '...' if longer."""
    return text[:max_len] + "..." if len(text) > max_len else text


def load_questions() -> list[dict]:
    """Load eval items from eval/questions.json. Each item has 'question' and optional 'expected_source'."""
    with open(QUESTIONS_PATH, "r") as f:
        return json.load(f)


def check_hit(returned_sources: list[str], expected_source) -> bool:
    """In-domain: at least one expected source in returned. Out-of-domain (expected_source None): hit iff returned is empty."""
    if expected_source is None:
        return not (returned_sources or [])
    if isinstance(expected_source, list):
        return any(exp in returned_sources for exp in expected_source)
    return expected_source in returned_sources

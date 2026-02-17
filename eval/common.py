"""Shared eval utilities: question loading and hit logic for retrieval and generation eval."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUESTIONS_PATH = ROOT / "eval" / "questions.json"


def load_questions() -> list[dict]:
    """Load eval items from eval/questions.json. Each item has 'question' and optional 'expected_source'."""
    with open(QUESTIONS_PATH, "r") as f:
        return json.load(f)


def check_hit(returned_sources: list[str], expected_source) -> bool:
    """
    True if returned sources match the expected criterion.

    - In-domain (expected_source is str or list): at least one expected source
      must appear in returned_sources.
    - Out-of-domain (expected_source is None): hit iff returned_sources is empty
      (no docs cited / no docs retrieved).
    """
    if expected_source is None:
        return not (returned_sources or [])
    if isinstance(expected_source, list):
        return any(exp in returned_sources for exp in expected_source)
    return expected_source in returned_sources

from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def get_txt_filenames() -> list[str]:
    """Return sorted list of .txt filenames in the data/ directory."""
    if not DATA_DIR.exists():
        return []
    return sorted(p.name for p in DATA_DIR.glob("*.txt"))


def load_text(filename: str) -> str:
    path = DATA_DIR / filename
    return path.read_text(encoding="utf-8")
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

def load_text(filename: str) -> str:
    path = DATA_DIR / filename
    return(path.read_text(encoding="utf-8"))

load_text("machine_learning_dirty.txt")
from pathlib import Path
from test_embeddings import embedding_size

# Path to data dir (project root / data)
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

# Read a .txt file from data/
def read_data_txt(filename: str) -> str:
    path = DATA_DIR / filename
    return path.read_text(encoding="utf-8")

# Read all .txt files in data/ one by one
for path in sorted(DATA_DIR.glob("*.txt")):
    text = path.read_text(encoding="utf-8")
    print(f"{path.name} - embedding size: {embedding_size(text)}")
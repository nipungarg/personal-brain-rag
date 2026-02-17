from .clean import clean_text
from .load import get_txt_filenames


def chunk_text(text: str, size=500, overlap=100) -> list[str]:
    chunks = []
    for i in range(0, len(text), size - overlap):
        chunks.append(text[i : i + size])
    return chunks


if __name__ == "__main__":
    filenames = get_txt_filenames()
    if filenames:
        text = clean_text(filenames[0])
        print(chunk_text(text)[0])
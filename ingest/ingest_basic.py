"""Ingest sanity check: print per-file chunk and embedding counts."""

from .load import get_txt_filenames
from .clean import clean_text
from .chunk_token import chunk_text
from .embedding import embed_chunks


if __name__ == "__main__":
    for filename in get_txt_filenames():
        text = clean_text(filename)
        chunks = chunk_text(text)
        n_emb = len(embed_chunks(chunks)) if chunks else 0
        print(f"{filename}: {len(chunks)} chunks, {n_emb} embeddings")
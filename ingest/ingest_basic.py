from .load import get_txt_filenames, load_text
from .test_embeddings import embedding_size


if __name__ == "__main__":
    for filename in get_txt_filenames():
        text = load_text(filename)
        print(f"{filename} - embedding size: {embedding_size(text)}")
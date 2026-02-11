from .clean import clean_text
from pathlib import Path
import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

DATA_DIR = Path(__file__).resolve().parent.parent / "data/llm_concepts_dirty.txt"

tokenizer=tiktoken.encoding_for_model("text-embedding-3-small")

def tik_token_len(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=150,
    chunk_overlap=50,
    length_function=tik_token_len,
    separators=["\n\n", "\n", " ", ""]
)

def chunk_text(text: str) -> list[str]:
    return text_splitter.split_text(text)

# print(chunk_text(clean_text(DATA_DIR))[1])
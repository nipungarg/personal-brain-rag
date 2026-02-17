import tiktoken
from langchain_text_splitters import RecursiveCharacterTextSplitter

tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")

def tik_token_len(text: str) -> int:
    return len(tokenizer.encode(text))

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100,
    length_function=tik_token_len,
    separators=["\n\n", "\n", " ", ""]
)

def chunk_text(text: str) -> list[str]:
    return text_splitter.split_text(text)
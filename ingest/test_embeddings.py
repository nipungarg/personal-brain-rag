import dotenv
import os
from pathlib import Path

from openai import OpenAI

dotenv.load_dotenv(Path(__file__).resolve().parent.parent / ".env")

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

def embedding_size(sentence: str) -> int:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=sentence,
    )
    return len(response.data[0].embedding)
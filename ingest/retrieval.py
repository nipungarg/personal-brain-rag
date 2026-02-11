import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import os
import chromadb
from dotenv import load_dotenv
from openai import OpenAI
from ingest.embedding import embed_query

load_dotenv(ROOT / ".env")
chroma_client = chromadb.PersistentClient(path=str(ROOT / "chroma_db"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
collection = chroma_client.get_collection("llm_concepts")

user_query = "How prompt engineering is different from tokenization?"

def get_relevant_context(user_query: str) -> str:
    query_response = embed_query(user_query)
    results = collection.query(
        query_embeddings=[query_response],
        n_results=3,
    )

    context = "\n---\n".join(results['documents'][0])
    return context

def generate_answer(user_query, context):
    system_prompt = """
    You are a helpful assistant. Use ONLY the provided context to answer the question. 
    If the answer is not in the context, say 'I do not have enough information in my vault.'
    Always cite the source if available in the metadata.
    """
    
    user_prompt = f"""
    Context:
    {context}
    
    Question: {user_query}
    
    Answer:"""

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.2,
        max_tokens=1000,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return response.choices[0].message.content

print(generate_answer(user_query, get_relevant_context(user_query)))
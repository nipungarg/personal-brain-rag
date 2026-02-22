# Personal Brain (RAG Vault)

A **Retrieval-Augmented Generation (RAG)** system that lets you query your personal documents and get grounded answers with citations. Documents live in `data/` as `.txt` files; the pipeline ingests them, builds searchable indexes, and at query time retrieves relevant chunks and sends them to an LLM to generate answers.

---

## What this codebase does

- **Ingest**: Loads `.txt` files from `data/`, cleans text, chunks by tokens (configurable size/overlap), and embeds chunks with OpenAI `text-embedding-3-small`.
- **Index**: Builds one of three backends:
  - **Chroma**: Persistent vector DB (default). Supports adaptive retrieval (dense + optional hybrid BM25, optional rerank), semantic response cache, and multiple collections (e.g. for chunk-config sweeps).
  - **Query**: In-memory retrieval over pre-loaded chunks (no separate index step; loads on first query).
  - **FAISS**: On-disk FAISS index + metadata; built once, then used for vector search.
- **Query**: Given a question, the chosen backend retrieves top-k chunks; a prompt is built and sent to **GPT-4o-mini**; the model returns an answer and cited sources.
- **Eval**: Retrieval and generation evaluation against `eval/questions.json` (expected sources per question); optional Chroma chunk-config sweep (`vault_small` / `vault_medium` / `vault_large`).

---

## Workflow (end-to-end)

```
data/*.txt  →  ingest (clean, chunk, embed)  →  index (Chroma / FAISS / or in-memory)
                                                          ↓
User question  →  ask.py (or eval)  →  retrieve top-k chunks  →  build prompt  →  LLM  →  answer + sources
```

1. **Put documents** in `data/` as `.txt` (e.g. `data/llm_concepts.txt`).
2. **Build an index** (Chroma or FAISS; query backend has no separate build step).
3. **Run** `ask.py` (or eval scripts) with the desired backend; Chroma can use cache and rerank.

---

## Setup

**Requirements:** Python 3.10+, OpenAI API key.

```bash
# Clone and enter project
cd personal-brain-rag

# Optional: virtualenv
python -m venv .venv && source .venv/bin/activate   # Linux/macOS
# or: .venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt

# API key (for embeddings + LLM)
echo "OPENAI_API_KEY=sk-..." > .env
```

---

## How to run

### 1. Ingest sanity check (optional)

Verifies loading, cleaning, chunking, and embedding for all `data/*.txt` files:

```bash
python -m ingest.ingest_basic
```

Example output: `llm_concepts.txt: 42 chunks, 42 embeddings` (and similar for other files).

---

### 2. Build the index

**Chroma (default collection `documents`):**

```bash
python -m chroma.store
# or programmatically: from chroma.store import build_index; build_index()
```

Use a **named collection** with custom chunk size/overlap (e.g. for experiments):

```bash
python -c "
from chroma.store import build_index_for_config
build_index_for_config('vault_small', chunk_size=300, chunk_overlap=50)
"
```

**FAISS:**

```bash
python -m faiss_rag.store
```

This writes the index and metadata under `faiss_rag/` (or paths set in `faiss_rag.client`).

**Query backend:** No build step. On first use it loads all `data/*.txt`, chunks and embeds in memory.

---

### 3. Ask questions (`ask.py`)

**Interactive (prompt for question):**

```bash
python ask.py
```

**With a question on the CLI:**

```bash
python ask.py --question "What is RAG?"
python ask.py -q "How does tokenization work?"
```

**Options:**

| Option | Description |
|--------|-------------|
| `-q`, `--question` | Question (omit for interactive prompt). |
| `--backend` | `chroma` (default), `query`, or `faiss`. |
| `--collection` | Chroma collection name (default: `documents`). |
| `--rerank K` | Chroma: retrieve K candidates, rerank to top-n; `0` = no rerank. |
| `--no-cache` | Disable semantic response cache (Chroma; cache is on by default). |
| `-n` | Number of chunks to retrieve (default: 4). |

**Examples:**

```bash
# Default: Chroma, cache on, 4 chunks
python ask.py -q "What is RAG?"

# Chroma, no cache, rerank with 20 candidates, 6 chunks
python ask.py -q "Define retrieval-augmented generation" --no-cache --rerank 20 -n 6

# Use FAISS index
python ask.py -q "Explain attention in transformers" --backend faiss

# Use in-memory query backend
python ask.py -q "What is tokenization?" --backend query
```

Output includes: cached (yes/no), timings (embed, retrieval, LLM), total tokens, cost, **ANSWER**, and **SOURCES**.

---

### 4. Evaluation

Eval uses `eval/questions.json` (questions + optional `expected_source`). Run from project root.

**Retrieval eval** (did we retrieve chunks from the expected source?):

```bash
python -m eval.evaluate_retrieval chroma
python -m eval.evaluate_retrieval chroma --collection vault_small
python -m eval.evaluate_retrieval faiss
python -m eval.evaluate_retrieval query
```

**Generation eval** (did the model cite the expected source?; uses LLM):

```bash
python -m eval.evaluate_generation chroma
python -m eval.evaluate_generation chroma --collection vault_small --rerank 20
python -m eval.evaluate_generation faiss
python -m eval.evaluate_generation query
```

Results print recall, timings, and (for generation) token/cost; logs go to `run_log.jsonl` (if configured).

---

### 5. Chunk-config sweep (Chroma)

Builds multiple Chroma collections (`vault_small`, `vault_medium`, `vault_large`) with different chunk sizes/overlaps, runs retrieval and generation eval for each, and prints a summary table:

```bash
python -m eval.chroma_chunk_sweep
```

**Options:**

| Option | Description |
|--------|-------------|
| `--skip-build` | Use existing vault_* collections (no rebuild). |
| `--retrieval-only` | Only run retrieval eval. |
| `--generation-only` | Only run generation eval. |

---

## Project layout (summary)

| Path | Purpose |
|------|---------|
| `data/*.txt` | Source documents. |
| `ingest/` | Load, clean, chunk (token-based), embed. |
| `chroma/` | Chroma index build, adaptive retrieval, cache, rerank, generation. |
| `query/` | Prompt builder, LLM client, semantic cache (SQLite), in-memory retrieve + generate. |
| `faiss_rag/` | FAISS index build, search, generation. |
| `eval/` | Questions JSON, retrieval/generation eval scripts, chunk sweep. |
| `ask.py` | CLI to ask questions (all backends). |
| `.env` | `OPENAI_API_KEY` (not committed). |

---

## Notes

- **Chroma** is the main backend: persistent, supports cache and rerank, and multiple collections for experiments.
- **Semantic cache** (Chroma): responses are cached by query embedding similarity (default threshold 0.90); paraphrased questions can reuse a cached answer.
- **Adaptive retrieval** (Chroma): keyword-heavy queries use hybrid (dense + BM25); otherwise dense with optional hybrid fallback if top score is low.

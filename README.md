# Personal Brain (RAG Vault)

A **Retrieval-Augmented Generation (RAG)** system that lets you query your personal documents and get grounded answers with citations. Documents live in `data/` as `.txt` files; the pipeline ingests them, builds searchable indexes, and at query time retrieves relevant chunks and sends them to an LLM to generate answers.

---

## What this codebase does

- **Stateless service**: The app holds no document state in memory. All vault data (vectors, cache, FAISS index) lives in persistent storage (configurable dirs). You can kill and restart the process anytime—cattle, not pets—with no data loss.
- **Ingest**: Loads `.txt` files from `data/`, cleans text, chunks by tokens (configurable size/overlap), and embeds chunks with OpenAI `text-embedding-3-small`.
- **Index**: Builds one of three backends:
  - **Chroma**: Persistent vector DB (default). Supports adaptive retrieval (dense + optional hybrid BM25, optional rerank), semantic response cache, and multiple collections (e.g. for chunk-config sweeps).
  - **Query**: In-memory retrieval over pre-loaded chunks (no separate index step; loads on first query).
  - **FAISS**: On-disk FAISS index + metadata; built once, then used for vector search.
- **Query**: Given a question, the chosen backend retrieves top-k chunks; a prompt is built and sent to **GPT-4o-mini**; the model returns an answer and cited sources.
- **Eval**: Retrieval and generation evaluation against `eval/questions.json` (expected sources per question); optional Chroma chunk-config sweep (`vault_small` / `vault_medium` / `vault_large`).

---

## Code flow (summary)

**Startup**  
Entry points (`app.py`, `ask.py`, `run_server.py`, eval scripts) call `config.ensure_root_path()` so the project root is on `sys.path`. `config.py` loads `.env` and exposes all settings (paths, models, retrieval params, etc.).

**Ingest (index build)**  
`chroma/store.py` → `ingest/load.get_txt_filenames()` for each `.txt` in `data/` → `ingest/clean.clean_text(filename)` (load + normalize whitespace, strip footers) → `ingest/chunk_token.chunk_text(text)` (token-based split, default size/overlap) → `ingest/metadata.get_metadata_for_chunks()` + `ingest/embedding.embed_chunks()` → `chroma/client.get_client()` (singleton `PersistentClient` on disk) → `add_documents()` into a Chroma collection. FAISS uses `faiss_rag/store` and writes index + metadata to disk.

**Query (Gradio app – main path)**  
1. User submits a question in `app.py` → `run_rag(question, use_rerank)`.
2. **Cache**: `query/cache.get_cached_response(question)` embeds the query (`ingest/embedding.embed_query`), compares to stored embeddings in SQLite via `utils/similarity.cosine_similarity`; if best similarity ≥ threshold, return cached `{answer, sources}` and stop (no retrieval/LLM).
3. **Retrieval**: `chroma/retrieve.get_relevant_chunks_adaptive(question, n_results, rerank_initial_k)`:
   - Uses `chroma/client.get_collection()` (default collection) and `embed_query(question)`.
   - If query is “keyword-heavy” (heuristic) → `get_relevant_chunks_hybrid()` (dense + BM25 RRF); else dense Chroma query, with optional hybrid fallback if top distance is above confidence threshold.
   - If `rerank_initial_k > 0`, fetches more candidates then `rerank_chunks()` (cross-encoder) and returns top `n_results`.
   - Returns list of chunk dicts `{source, chunk_index, id, text}`.
4. **Prompt**: `query/prompt.build_prompt(question, chunks)` formats context with `format_context(chunks)` and adds instructions + “SOURCES: …” format.
5. **LLM**: `query/llm.complete_rag_stream(prompt)` uses OpenAI singleton (`query/llm.get_client()`), streams chunks, accumulates content, then `query/prompt.parse_response(content)` → `{answer, sources}`; attaches usage/cost. App yields streamed deltas then final answer/sources/time/cost.
6. **Cache write**: On final result, `query/cache.set_cached_response(question, answer, sources)` stores query embedding + answer + sources in SQLite.

**Query (ask.py CLI)**  
Same idea: backend `chroma` → `chroma/generate.generate_answer()` (optional cache, then `get_relevant_chunks_adaptive` → `build_prompt` → `query/llm.complete_rag` non-streaming); backends `query` / `faiss` use `query/generate` or `faiss_rag/generate` (in-memory or FAISS retrieval, same prompt + LLM).

**Server lifecycle (run_server.py)**  
`run_server.py` builds a FastAPI app with a lifespan: on startup, `state.register_signal_handlers()` (SIGTERM/SIGINT + atexit); on shutdown, `state.shutdown()` which calls `chroma/client.close_client()` and `query/llm.reset_client()`. Gradio is mounted at `/`. Uvicorn serves the app; no document state is kept in process—all data is in Chroma dir, cache DB, or FAISS files.

---

## Workflow (end-to-end)

```
data/*.txt  →  ingest (clean, chunk, embed)  →  index (Chroma / FAISS / or in-memory)
                                                          ↓
User question  →  app.py / ask.py / eval  →  retrieve top-k  →  build prompt  →  LLM  →  answer + sources
```

1. **Put documents** in `data/` as `.txt`.
2. **Build an index** (Chroma or FAISS).
3. **Run** the Gradio app (`app.py`), CLI (`ask.py`), or eval scripts.

---

## Setup

**Requirements:** Python 3.9+, OpenAI API key. All non-secret settings live in `config.py` (env with defaults); entry points call `config.ensure_root_path()` so the project root is on `sys.path`.

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

**Query backend:** No build step; loads and embeds on first query.

---

### 3. Gradio app (`app.py`)

Web UI with streaming answers, semantic cache, and optional rerank:

```bash
python app.py
# or with auto-reload: gradio app.py
```

Open the URL (e.g. http://127.0.0.1:7860). Use the **Use rerank** checkbox to rerank retrieval. Repeat/similar questions are served from cache ($0 cost).

---

### 4. Long-lived server (`run_server.py`)

Run the app as a proper service (FastAPI + Gradio) with graceful shutdown. All vault data stays on disk; the process is stateless—kill and restart anytime.

```bash
python run_server.py
# or: uvicorn run_server:app --host 0.0.0.0 --port 7860
```

Host and port are set by `SERVER_HOST` and `SERVER_PORT` in `config` / `.env` (defaults: `0.0.0.0`, `7860`). On SIGTERM/SIGINT the server drains and closes Chroma/OpenAI client refs cleanly.

---

### 5. Ask questions (`ask.py`)

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

### 6. Evaluation

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

### 7. Chunk-config sweep (Chroma)

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
| `config.py` | Loads `.env`; exposes all settings and `ensure_root_path()`. |
| `state.py` | Process lifecycle only: graceful shutdown, close Chroma/OpenAI refs. |
| `data/*.txt` | Source documents. |
| `ingest/` | Load, clean, chunk (token-based), embed. |
| `chroma/` | Chroma index build, adaptive retrieval, rerank, generation. |
| `query/` | Prompt builder, LLM client, semantic cache (SQLite), in-memory retrieve + generate. |
| `faiss_rag/` | FAISS index build, search, generation. |
| `eval/` | Questions JSON, retrieval/generation eval scripts, chunk sweep. |
| `utils/` | Shared helpers (e.g. `similarity.cosine_similarity` for cache and in-memory retrieval). |
| `app.py` | Gradio UI (streaming, cache, rerank; Chroma only). |
| `ask.py` | CLI to ask questions (all backends). |
| `run_server.py` | Long-lived service: FastAPI + Gradio, graceful shutdown. |
| `.env` | `OPENAI_API_KEY` and optional overrides (not committed). |

---

## Notes

- **Chroma** is the main backend: persistent, supports cache and rerank, and multiple collections for experiments.
- **Semantic cache** (Chroma): responses are cached by query embedding similarity (default threshold 0.90); paraphrased questions can reuse a cached answer.
- **Adaptive retrieval** (Chroma): keyword-heavy queries use hybrid (dense + BM25); otherwise dense with optional hybrid fallback if top score is low.
- **Stateless + graceful shutdown**: Vault data lives only in persistent storage (Chroma dir, cache DB, FAISS files). `run_server.py` registers signal handlers and closes client refs on shutdown so the service can be stopped and restarted cleanly.

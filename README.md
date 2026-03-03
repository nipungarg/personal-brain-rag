# Personal Brain (RAG Vault)

RAG over your documents: ingest `.txt` from `data/`, build a vector index (Chroma or FAISS), then ask questions and get answers with citations. Uses OpenAI for embeddings and GPT-4o-mini for generation.

**Highlights:** Stateless (vault on disk; kill/restart anytime), semantic cache, optional rerank, structured logging with request IDs, deployable to Render.

---

## Quick start

```bash
cd personal-brain-rag
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
echo "OPENAI_API_KEY=sk-..." > .env
```

1. Put `.txt` files in `data/`.
2. Build the index: `python -m chroma.store`
3. Run the app: `python app.py` or `python run_server.py` (long-lived server)

Open the URL (e.g. http://127.0.0.1:7860), ask a question. First request can be slow (Chroma/tiktoken load); leave **Use rerank** off for the first question.

---

## How to run

| Command | Purpose |
|--------|--------|
| `python app.py` | Gradio UI (streaming, cache, rerank). |
| `python run_server.py` | FastAPI + Gradio server (graceful shutdown, use for production). |
| `python ask.py` | CLI: `ask.py -q "What is RAG?"` (supports `--backend chroma\|query\|faiss`, `--no-cache`, `--rerank K`, `-n`). |
| `python -m chroma.store` | Build Chroma index (default collection `documents`). |
| `python -m faiss_rag.store` | Build FAISS index. |
| `python -m ingest.ingest_basic` | Sanity check: load, clean, chunk, embed all `data/*.txt`. |

**Server:** `run_server.py` reads `PORT` when set (e.g. Render); otherwise uses `SERVER_PORT` or 7860. Health check: `GET /health`.

---

## Deploy to Render

1. Push the repo; connect it in [Render](https://render.com) as a **Blueprint** (uses `render.yaml`).
2. In the Web Service, set **OPENAI_API_KEY** as a **Secret**.
3. Start command is already: `uvicorn run_server:app --host 0.0.0.0 --port $PORT`.

For free tier: commit `chroma_db/` (remove from `.gitignore`) so the app has an index without a persistent disk. First request may be slow (90s retrieval timeout; increase with `RETRIEVAL_TIMEOUT` if needed). See `docs/DEPLOY_RENDER.md` for details.

---

## Config and env

All settings are in `config.py`; values come from `.env` (see `.env.example`). Required: **OPENAI_API_KEY**. Common overrides:

| Env | Default | Purpose |
|-----|--------|--------|
| `LOG_LEVEL` | `INFO` | Logging level. |
| `OPENAI_TIMEOUT` | `120` | Timeout (seconds) for OpenAI API calls. |
| `RETRIEVAL_TIMEOUT` | `90` | Timeout for retrieval (avoids long hangs on first Chroma load). |
| `CHROMA_PERSIST_DIR` | `chroma_db` | Chroma data directory. |
| `SERVER_PORT` | `7860` | Port when `PORT` is not set (e.g. Render sets `PORT`). |

Logs: JSON lines to stderr (and optionally `LOG_FILE`). Each request has a short **request_id**; logs include retrieval/LLM latency, cache hit, and errors.

---

## Eval

Uses `eval/questions.json` (questions + optional `expected_source`).

```bash
python -m eval.evaluate_retrieval chroma
python -m eval.evaluate_generation chroma
python -m eval.chroma_chunk_sweep   # vault_small / vault_medium / vault_large
```

---

## Project layout

| Path | Purpose |
|------|--------|
| `config.py` | Loads `.env`; exposes settings and `ensure_root_path()`. |
| `state.py` | Graceful shutdown (close Chroma/OpenAI refs). |
| `app.py` | Gradio UI (Chroma only). |
| `ask.py` | CLI (chroma / query / faiss). |
| `run_server.py` | FastAPI + Gradio server. |
| `chroma/` | Index build, adaptive retrieval, rerank, generation. |
| `query/` | Prompt, LLM, semantic cache (SQLite), in-memory retrieval. |
| `ingest/` | Load, clean, chunk, embed. |
| `faiss_rag/` | FAISS index and retrieval. |
| `eval/` | Eval scripts and chunk sweep. |
| `utils/` | Shared helpers (e.g. `similarity`, `logging_config`). |

---

## Notes

- **Backends:** Chroma (default; persistent, cache, rerank, multiple collections), FAISS (on-disk index), or in-memory **query** (no index build).
- **Semantic cache:** Query similarity ≥ threshold (default 0.90) returns a cached response; paraphrases can hit cache.
- **Adaptive retrieval:** Keyword-heavy queries use hybrid (dense + BM25); otherwise dense with optional hybrid fallback.
- **Stateless:** All vault data is on disk; the process only holds client refs. Shutdown closes them cleanly.

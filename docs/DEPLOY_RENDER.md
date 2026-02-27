# Deploy to Render

Deploy the Personal Brain RAG app (FastAPI + Gradio) as a **Web Service** on [Render](https://render.com).

## Prerequisites

- A [Render](https://render.com) account (free tier works).
- Your repo pushed to GitHub/GitLab (Render connects to the repo).
- **OpenAI API key** (required for embeddings and LLM).

## Option A: One-click with Blueprint (recommended)

1. **Connect repo**
   - In [Render Dashboard](https://dashboard.render.com), click **New** → **Blueprint**.
   - Connect your Git provider and select the `personal-brain-rag` repo.
   - Render will detect `render.yaml` in the repo root.

2. **Secrets**
   - When the Blueprint is created, open the **Web Service** (e.g. `personal-brain-rag`).
   - Go to **Environment** and add:
     - **OPENAI_API_KEY** → your key (mark as **Secret**).

3. **Deploy**
   - Save; Render will build and deploy. The app will be at `https://<service-name>.onrender.com`.

4. **Index (first time)**
   - The app needs a Chroma index. Options:
     - **Without persistent disk**: Build the index locally, then run the app locally for testing; on Render the instance is ephemeral, so each deploy starts with no index unless you add a disk or external storage.
     - **With persistent disk** (paid): Add a disk, set `CHROMA_PERSIST_DIR` to a path on the disk (e.g. `/data/chroma_db`), then run an ingest step (e.g. via a one-off job or SSH) to populate the index.

## Option B: Manual Web Service (no `render.yaml`)

1. **New Web Service**
   - Dashboard → **New** → **Web Service**.
   - Connect the repo and select it.

2. **Settings**
   - **Name**: e.g. `personal-brain-rag`.
   - **Region**: e.g. Oregon (or your choice).
   - **Branch**: `main` (or your default).
   - **Runtime**: **Python 3**.
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `python run_server.py`
   - **Instance type**: Free (or paid for persistent disk).

3. **Environment**
   - Add **OPENAI_API_KEY** (Secret) with your OpenAI key.
   - Render sets **PORT** automatically; the app uses it via `config.SERVER_PORT` (which reads `PORT` when present).

4. **Deploy**
   - Click **Create Web Service**. Render will build and deploy.

## Port and URL

- Render assigns a **PORT** (e.g. 10000). The app reads it via `PORT` in `config.py` and binds `0.0.0.0:PORT`.
- Your app URL: `https://<service-name>.onrender.com`.

## Data persistence (optional)

- **Free tier**: No persistent disk; the filesystem is ephemeral. Chroma DB, cache, and FAISS data are lost on redeploy or restart. Use for trying the app; for real use add a disk or external storage.
- **Starter (or higher) + Persistent Disk**:
  - In the Web Service, add a **Persistent Disk** (e.g. mount at `/data`, 1 GB).
  - Set env vars so data lives on the disk, e.g.:
    - `CHROMA_PERSIST_DIR=/data/chroma_db`
    - `CACHE_DB=/data/cache.db`
    - `FAISS_DB=/data/faiss_db`
  - Run ingest (e.g. a one-off **Background Worker** or **Shell** that runs `python -m chroma.store`) so the index is built on the disk. After that, the Web Service will use the same index across deploys.

## Health check (optional)

Render can ping a path to check if the app is up. The Gradio app is mounted at `/`. You can add a small health route in `run_server.py` (e.g. FastAPI `@app.get("/health")` returning `{"status": "ok"}`) and set Render’s **Health Check Path** to `/health`.

## Troubleshooting

- **Build fails**: Ensure `requirements.txt` is at repo root and `PYTHON_VERSION` (e.g. `3.11.11`) is set in the service env if you need a specific version.
- **App not loading**: Check **Logs** in the Render dashboard; structured logs go to stderr and are visible there.
- **No index / empty answers**: Without a persistent disk, the Chroma index must be built after each deploy (e.g. run ingest in a one-off job or locally and use a disk or S3 for Chroma if you add that support).

#!/usr/bin/env python3
"""Long-lived RAG service: FastAPI + Gradio, graceful shutdown.

The app is stateless; all vault data lives in persistent storage (Chroma,
cache DB, FAISS). Run as cattle—kill and restart anytime without data loss.
"""

import os
import sys

# Log immediately so Render shows we started (helps debug "no open ports")
print("run_server: process started", flush=True)
sys.stdout.flush()

from config import ensure_root_path, SERVER_HOST, SERVER_PORT
ensure_root_path()

from utils.logging_config import configure_logging
configure_logging()

from contextlib import asynccontextmanager

from fastapi import FastAPI
import gradio as gr

from state import register_signal_handlers, shutdown

# Import demo after path setup so app module resolves (can be slow on first load)
print("run_server: loading Gradio app...", flush=True)
sys.stdout.flush()
from app import demo
print("run_server: Gradio app loaded", flush=True)
sys.stdout.flush()


@asynccontextmanager
async def lifespan(app: FastAPI):
    register_signal_handlers()
    yield
    shutdown()


app = FastAPI(title="Personal Brain RAG", lifespan=lifespan)


@app.get("/health")
def health():
    """Liveness/readiness for Render and load balancers."""
    return {"status": "ok"}


app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    # Render/Heroku set PORT; read at runtime so the service binds to the correct port
    port = int(os.environ.get("PORT", SERVER_PORT))
    print(f"run_server: binding to 0.0.0.0:{port}", flush=True)
    sys.stdout.flush()
    uvicorn.run(
        app,  # pass app object so port binding happens in this process
        host="0.0.0.0",
        port=port,
        reload=False,
    )

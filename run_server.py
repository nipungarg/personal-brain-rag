#!/usr/bin/env python3
"""Long-lived RAG service: FastAPI + Gradio, graceful shutdown.

The app is stateless; all vault data lives in persistent storage (Chroma,
cache DB, FAISS). Run as cattle—kill and restart anytime without data loss.
"""

from config import ensure_root_path, SERVER_HOST, SERVER_PORT
ensure_root_path()

from utils.logging_config import configure_logging
configure_logging()

from contextlib import asynccontextmanager

from fastapi import FastAPI
import gradio as gr

from state import register_signal_handlers, shutdown

# Import demo after path setup so app module resolves
from app import demo


@asynccontextmanager
async def lifespan(app: FastAPI):
    register_signal_handlers()
    yield
    shutdown()


app = FastAPI(title="Personal Brain RAG", lifespan=lifespan)

app = gr.mount_gradio_app(app, demo, path="/")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "run_server:app",
        host=SERVER_HOST,
        port=SERVER_PORT,
        reload=False,
    )

"""Process lifecycle only: graceful shutdown and cleanup. No document/vault state.

The application is stateless. All document vault data lives in the persistent
storage layer (Chroma dir, cache DB, FAISS files). This module only closes
client refs on shutdown. The service is cattle, not pets: kill and restart anytime.
"""

import atexit
import signal
import sys

_is_shutting_down = False


def shutdown() -> None:
    """Close long-lived resources and set shutdown flag. Safe to call multiple times."""
    global _is_shutting_down
    if _is_shutting_down:
        return
    _is_shutting_down = True
    try:
        from chroma.client import close_client
        close_client()
    except Exception:
        pass
    try:
        from query import llm
        if hasattr(llm, "reset_client"):
            llm.reset_client()
    except Exception:
        pass


def _handle_signal(signum, frame):
    shutdown()
    sys.exit(0)


def register_signal_handlers() -> None:
    """Register SIGTERM and SIGINT for graceful shutdown."""
    signal.signal(signal.SIGTERM, _handle_signal)
    signal.signal(signal.SIGINT, _handle_signal)
    atexit.register(shutdown)

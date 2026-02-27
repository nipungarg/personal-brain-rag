"""Structured logging: JSON lines, request_id via contextvars, and optional file output."""

import json
import logging
import os
import sys
from contextvars import ContextVar
from typing import Any, Optional

# Request ID for the current request (set at entry points; included in all logs in that context).
_request_id: ContextVar[Optional[str]] = ContextVar("request_id", default=None)


def set_request_id(value: Optional[str]) -> None:
    _request_id.set(value)


def get_request_id() -> Optional[str]:
    return _request_id.get()


# Standard LogRecord attribute names (so we don't duplicate them as "extra" in JSON).
_RECORD_ATTRS = frozenset({
    "name", "msg", "args", "levelname", "levelno", "pathname", "filename",
    "module", "lineno", "funcName", "created", "msecs", "relativeCreated",
    "thread", "threadName", "process", "processName", "message", "exc_info",
    "exc_text", "stack_info", "taskName", "getMessage",
})


class JsonFormatter(logging.Formatter):
    """Format each log record as a single JSON object (one line)."""

    def format(self, record: logging.LogRecord) -> str:
        request_id = getattr(record, "request_id", None) or get_request_id()
        out = {
            "timestamp": self.formatTime(record, self.datefmt),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if request_id is not None:
            out["request_id"] = request_id
        # Include any extra keys passed to log(..., extra={...})
        for key, value in record.__dict__.items():
            if key not in _RECORD_ATTRS and value is not None:
                if isinstance(value, (str, int, float, bool)) or value is None:
                    out[key] = value
                else:
                    out[key] = str(value)
        if record.exc_info:
            out["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(out, default=str)


_configured = False


def configure_logging(
    level: Optional[str] = None,
    stream: Any = sys.stderr,
    log_file: Optional[str] = None,
) -> None:
    """Configure root logger with structured JSON format. Idempotent."""
    global _configured
    if _configured:
        return
    _configured = True

    from config import LOG_FILE as CONFIG_LOG_FILE, LOG_LEVEL as CONFIG_LOG_LEVEL, ROOT
    log_level_str = (level or os.environ.get("LOG_LEVEL") or CONFIG_LOG_LEVEL or "INFO").upper()
    numeric = getattr(logging, log_level_str, logging.INFO)
    root = logging.getLogger()
    root.setLevel(numeric)
    formatter = JsonFormatter()

    handler = logging.StreamHandler(stream)
    handler.setLevel(numeric)
    handler.setFormatter(formatter)
    root.addHandler(handler)

    file_path = log_file or CONFIG_LOG_FILE
    if file_path and not os.path.isabs(file_path):
        file_path = str(ROOT / file_path)
    if file_path:
        try:
            fh = logging.FileHandler(file_path, encoding="utf-8")
            fh.setLevel(numeric)
            fh.setFormatter(formatter)
            root.addHandler(fh)
        except OSError:
            pass

    # Reduce noise from third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """Return a logger for the given name. Call configure_logging() first from entry points."""
    return logging.getLogger(name)

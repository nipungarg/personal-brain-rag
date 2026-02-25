import json
from datetime import datetime

from config import LOG_FILE


def log_event(event_type: str, data: dict):
    record = {
        "timestamp": datetime.utcnow().isoformat(),
        "event": event_type,
        "data": data,
    }
    with open(LOG_FILE, "a") as f:
        f.write(json.dumps(record) + "\n")

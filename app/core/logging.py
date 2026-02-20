import json
import logging
import sys
from datetime import datetime, timezone

_BASE_LOG_KEYS = set(logging.makeLogRecord({}).__dict__.keys())


def _safe_json_value(value):
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, dict)):
        return value
    return str(value)


def _json_formatter(record: logging.LogRecord) -> str:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "level": record.levelname,
        "logger": record.name,
        "message": record.getMessage(),
    }
    for key, value in record.__dict__.items():
        if key in _BASE_LOG_KEYS or key in {"message", "asctime"}:
            continue
        payload[key] = _safe_json_value(value)
    if record.exc_info:
        payload["exc_info"] = logging.Formatter().formatException(record.exc_info)
    if hasattr(record, "request_id"):
        payload["request_id"] = record.request_id
    return json.dumps(payload, ensure_ascii=True)


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        return _json_formatter(record)


def configure_logging(level: str = "INFO") -> None:
    root = logging.getLogger()
    root.setLevel(level.upper())

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonLogFormatter())

    root.handlers.clear()
    root.addHandler(handler)

    # keep noisy dependency logs down so app logs are easier to read
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("posthog").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)

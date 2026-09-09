# 1. Configure a JSON Formatter (python-json-logger or small custom logging.Formatter)
# 2. Every Log Line carries:
# - Timestamp
# - Log Level
# - Logger Name
# - Message
# - Correlation ID
# 3. Generate a Correlation ID (uuid4) in FastAPI Middleware and attach it via `contextvars`, return it as an `X-Request-ID` Header.
# 4. Use levels correctly and prove you understand them:
# - DEBUG : feature vector ·
# - WARNING : input outside the training range (trip_distance > 100 ).
# - INFO : prediction served with latency ·
# - ERROR : model load failure, validation rejection.
# 5. Delete every `print()` from `src/`

# ---> Imports
import logging
import json
import sys
from contextvars import ContextVar
from datetime import datetime, timezone
from typing import Any
# from uuid import uuid4

# ---> Context Variable for Correlation ID
# Define a context variable to hold the correlation ID for each request
correlation_id_var: ContextVar[str] = ContextVar("correlation_id", default="-")


class JSONFormatter(logging.Formatter):
    """
    Custom JSON Formatter for logging.
    Each log line will be formatted as a JSON object with the following fields:
    - timestamp
    - level
    - logger
    - message
    - correlation_id
    """

    def format(self, record: logging.LogRecord) -> str:
        log_record: dict[str, Any] = {
            # self.formatTime(record, self.datefmt)
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "correlation_id": correlation_id_var.get(),
        }
        if hasattr(record, "extra") and isinstance(record.extra, dict):
            log_record.update(record.extra)

        return json.dumps(log_record, ensure_ascii=False)


def setup_logging(name: str) -> None:
    """
    Set up logging configuration with JSON formatting.
    Args:
        level: Logging level (default: INFO)
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(JSONFormatter())
        logger.handlers.clear()  # Clear existing handlers
        logger.addHandler(handler)
        logger.setLevel(logging.DEBUG)
        logger.propagate = (
            False  # Prevent log messages from being propagated to the root logger
        )

    return logger


# def get_correlation_id() -> str:
#     """
#     Retrieve the current correlation ID from the context variable.
#     If not set, generate a new UUID4 and set it in the context.
#     Returns:
#         The current correlation ID as a string.
#     """
#     correlation_id = correlation_id_var.get()
#     if not correlation_id:
#         correlation_id = str(uuid4())
#         correlation_id_var.set(correlation_id)
#     return correlation_id


# def set_correlation_id(correlation_id: str) -> str:
#     """
#     Set the correlation ID in the context variable.
#     Args:
#         correlation_id: The correlation ID to set.
#     Returns:
#         The correlation ID that was set.
#     """
#     correlation_id = correlation_id or str(uuid4())
#     correlation_id_var.set(correlation_id)
#     return correlation_id

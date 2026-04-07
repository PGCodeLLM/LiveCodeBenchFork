"""Centralized logging for lcb_runner.

Replaces print() calls with proper logging to avoid BlockingIOError (errno 11)
when stdout buffer is full under heavy I/O with many parallel processes.
"""

import io
import logging
import os
import sys

# Force line-buffered stderr so tqdm progress and log messages flush immediately,
# even in non-TTY environments (e.g. Docker, piped output).
if hasattr(sys.stderr, "buffer"):
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, line_buffering=True)
if hasattr(sys.stdout, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, line_buffering=True)

# Default max length for truncated log messages
_MAX_LOG_STR_LEN = 500

# Read log level from env: LCB_LOG_LEVEL=DEBUG|INFO|WARNING|ERROR (default: INFO)
_LOG_LEVEL = getattr(logging, os.environ.get("LCB_LOG_LEVEL", "INFO").upper(), logging.INFO)


def truncate(s, max_len: int = _MAX_LOG_STR_LEN) -> str:
    """Truncate a string to max_len, showing head and tail with '...' in between."""
    text = str(s)
    if len(text) <= max_len:
        return text
    half = max_len // 2
    return text[:half] + f"...<truncated {len(text) - max_len} chars>..." + text[-half:]


def setup_logger(name: str = "lcb_runner", level: int = _LOG_LEVEL) -> logging.Logger:
    """Get or create a logger with a non-blocking stderr handler.

    Uses stderr instead of stdout because:
    - stdout may be redirected/piped (triggering non-blocking I/O)
    - stderr is line-buffered by default and less likely to block

    The handler is configured to silently drop messages rather than
    raise BlockingIOError when the buffer is full.
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S",
            )
        )
        # Override emit to swallow BlockingIOError
        _original_emit = handler.emit

        def _safe_emit(record):
            try:
                _original_emit(record)
            except BlockingIOError:
                pass

        handler.emit = _safe_emit
        logger.addHandler(handler)
        logger.setLevel(level)

    return logger


# Module-level default logger
logger = setup_logger()

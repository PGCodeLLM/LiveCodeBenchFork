"""Centralized logging for lcb_runner.

Replaces print() calls with proper logging to avoid BlockingIOError (errno 11)
when stdout buffer is full under heavy I/O with many parallel processes.
"""

import logging
import sys


def setup_logger(name: str = "lcb_runner", level: int = logging.INFO) -> logging.Logger:
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

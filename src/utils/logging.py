"""Shared logging utility for the Somniloquy project.

Usage:
    from src.utils.logging import get_logger
    logger = get_logger(__name__)
    logger.info("Processing tensor of shape %s", x.shape)
"""

import logging

_LOG_FORMAT = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"


def get_logger(name: str) -> logging.Logger:
    """Return a logger configured at INFO level with a single StreamHandler.

    Calling this function multiple times with the same name returns the same
    logger instance and does NOT duplicate handlers (idempotent).

    Args:
        name: Logger name, typically __name__ of the calling module.

    Returns:
        Configured logging.Logger instance.
    """
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter(_LOG_FORMAT))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

"""TDD tests for src.utils.logging — RED phase."""

import logging


def test_get_logger_returns_logger():
    from src.utils.logging import get_logger
    logger = get_logger("test_basic")
    assert isinstance(logger, logging.Logger)


def test_get_logger_info_level():
    from src.utils.logging import get_logger
    logger = get_logger("test_level")
    assert logger.level == logging.INFO


def test_get_logger_idempotent_handlers():
    from src.utils.logging import get_logger
    l1 = get_logger("test_idem")
    l2 = get_logger("test_idem")
    assert l1 is l2
    assert len(l1.handlers) == 1


def test_get_logger_has_stream_handler():
    from src.utils.logging import get_logger
    logger = get_logger("test_stream")
    assert any(isinstance(h, logging.StreamHandler) for h in logger.handlers)


def test_get_logger_format():
    from src.utils.logging import get_logger
    logger = get_logger("test_format")
    handler = logger.handlers[0]
    fmt = handler.formatter._fmt
    assert "%(asctime)s" in fmt
    assert "%(levelname)s" in fmt
    assert "%(name)s" in fmt
    assert "%(message)s" in fmt

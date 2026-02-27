"""Structured logging configuration using structlog."""

from __future__ import annotations

import logging
import sys
from typing import Protocol

import structlog


class Logger(Protocol):
    """Minimal logger protocol for basedpyright compatibility.

    structlog.stdlib.BoundLogger methods return Any, which basedpyright
    treats as Unknown.  This protocol gives callers a concrete type.
    """

    def debug(self, event: str | None = None, **kw: object) -> None: ...
    def info(self, event: str | None = None, **kw: object) -> None: ...
    def warning(self, event: str | None = None, **kw: object) -> None: ...
    def error(self, event: str | None = None, **kw: object) -> None: ...
    def critical(self, event: str | None = None, **kw: object) -> None: ...
    def exception(self, event: str | None = None, **kw: object) -> None: ...


def setup_logging(*, verbose: bool = False) -> None:
    """Configure structlog for the application.

    Args:
        verbose: When *True*, set log level to DEBUG; otherwise INFO.
    """
    level = logging.DEBUG if verbose else logging.INFO

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=level,
        force=True,
    )

    shared_processors: list[structlog.types.Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso", utc=False),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    structlog.configure(
        processors=[
            *shared_processors,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty()),
        ],
    )

    root = logging.getLogger()
    for handler in root.handlers:
        handler.setFormatter(formatter)

    if not verbose:
        # Reduce noisy transport logs in normal mode.
        logging.getLogger("uvicorn.access").setLevel(logging.WARNING)
        logging.getLogger("httpx").setLevel(logging.WARNING)
        logging.getLogger("httpcore").setLevel(logging.WARNING)
        logging.getLogger("websockets").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> Logger:
    """Return a bound structlog logger."""
    return structlog.stdlib.get_logger(name)  # type: ignore[return-value]

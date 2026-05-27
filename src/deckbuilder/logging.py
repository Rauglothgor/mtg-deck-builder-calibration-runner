"""Logging helpers."""

import logging

import structlog


def configure_logging(level: str = "INFO") -> None:
    """Configure structlog for JSON-friendly console output."""
    logging.basicConfig(level=getattr(logging, level.upper(), logging.INFO))
    structlog.configure(processors=[structlog.processors.JSONRenderer()])

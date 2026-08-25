"""Logging config. Called once from main.py at startup."""

from __future__ import annotations

import logging

from app.config import settings


def configure_logging() -> None:
    logging.basicConfig(
        level=getattr(logging, settings.LOG_LEVEL.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-5.5s [%(name)s] %(message)s",
    )

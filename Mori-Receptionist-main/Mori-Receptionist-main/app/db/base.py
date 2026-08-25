"""SQLAlchemy DeclarativeBase + shared helpers for receptionist models.

Every model in db/models/ inherits from `Base`. Shared utilities live here so
the same import style works across model files:

    from app.db.base import Base, utcnow
"""

from datetime import datetime, timezone

from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    """Shared base class for all receptionist tables."""

    pass


def utcnow() -> datetime:
    """Timezone-aware UTC now. Use this instead of `datetime.utcnow()`,
    which produces a naive datetime and silently misbehaves with Postgres
    `TIMESTAMPTZ` columns."""
    return datetime.now(timezone.utc)

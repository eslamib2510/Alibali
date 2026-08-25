"""Database engine + session factory for the receptionist DB.

Uses sync SQLAlchemy (matches the rest of AlaBali backend). Two URLs come
from env via settings:

  - RECEPTIONIST_DATABASE_URL          → pooled, used by the app
  - RECEPTIONIST_DATABASE_DIRECT_URL   → direct, used by Alembic only

The app should only ever import `SessionLocal` / `get_session()` from here.
"""

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings


def _psycopg_url(url: str) -> str:
    """Force SQLAlchemy to use psycopg (v3) instead of the default psycopg2.
    We ship psycopg[binary] in pyproject.toml; adding the explicit driver
    prefix avoids "ModuleNotFoundError: psycopg2" when the raw postgresql://
    scheme is stored in env."""
    if url.startswith("postgresql://"):
        return url.replace("postgresql://", "postgresql+psycopg://", 1)
    return url


# `pool_pre_ping` issues a cheap `SELECT 1` before reusing a connection from
# the pool. Cheap insurance against Neon's serverless compute parking idle
# connections and dropping them.
engine = create_engine(
    _psycopg_url(settings.RECEPTIONIST_DATABASE_URL),
    pool_pre_ping=True,
    future=True,
)

SessionLocal = sessionmaker(bind=engine, expire_on_commit=False)


@contextmanager
def get_session(tenant_id=None) -> Iterator[Session]:
    """Yield a session, commit on success, rollback on exception, always close.

        with get_session() as db:
            db.add(obj)
            # auto-commits at the end of the block

    Pass `tenant_id` for any session that touches a table with row-level
    security (currently `knowledge`):

        with get_session(tenant_id=tenant.id) as db:
            ...

    That sets the `app.tenant_id` GUC the RLS policy reads, so Postgres
    filters every query to that tenant even if a `WHERE tenant_id` is
    forgotten. Without it, RLS-protected tables return nothing — which is the
    safe direction to fail.

    `set_config(..., true)` is the function form of `SET LOCAL`: it accepts a
    bind parameter (plain `SET LOCAL` does not) and is scoped to the current
    transaction, so the value can never leak to the next checkout of this
    pooled connection.
    """
    db = SessionLocal()
    try:
        if tenant_id is not None:
            db.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"),
                {"tid": str(tenant_id)},
            )
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()

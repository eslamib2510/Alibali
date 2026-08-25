"""Alembic env — wires our models + Neon connection.

Important: we use RECEPTIONIST_DATABASE_DIRECT_URL (not the pooled one).
Neon's pooler (PgBouncer in transaction mode) doesn't play nice with the
transactional DDL Alembic runs.
"""

from __future__ import annotations

from logging.config import fileConfig
from pathlib import Path

from alembic import context
from dotenv import load_dotenv
from sqlalchemy import engine_from_config, pool

# env.py lives at app/db/migrations/env.py — project root is 3 levels up.
# Load .env explicitly before touching settings.
_PROJECT_ROOT = Path(__file__).resolve().parents[3]
load_dotenv(_PROJECT_ROOT / ".env")

# Make sure all models are imported so `Base.metadata` is fully populated
# before autogenerate inspects it. Importing the package's __init__ pulls
# every model in.
from app.config import settings  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db import models as _models  # noqa: F401, E402

config = context.config

# Inject the direct DB URL from settings (sourced from env via pydantic).
db_url = settings.RECEPTIONIST_DATABASE_DIRECT_URL or settings.RECEPTIONIST_DATABASE_URL
if not db_url:
    raise RuntimeError(
        "RECEPTIONIST_DATABASE_DIRECT_URL (or RECEPTIONIST_DATABASE_URL) must be set "
        "in the environment for Alembic to run."
    )
# Force psycopg3 driver — we ship psycopg[binary], not psycopg2.
if db_url.startswith("postgresql://"):
    db_url = db_url.replace("postgresql://", "postgresql+psycopg://", 1)
config.set_main_option("sqlalchemy.url", db_url)

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode (emit SQL only, no DB connection)."""
    context.configure(
        url=db_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode (connect to DB, apply)."""
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()

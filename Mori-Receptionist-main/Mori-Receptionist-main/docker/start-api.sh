#!/bin/sh
# API entrypoint: apply any pending Alembic migrations, then start uvicorn.
# If migrations fail the container exits — we never serve traffic against a
# half-migrated schema.
#
# Only the api service runs this. Worker uses its own command and skips
# migrations (avoids a race where two containers try to migrate on startup).

set -e

echo "[start] applying migrations..."
alembic upgrade head

echo "[start] launching uvicorn..."
exec uvicorn app.main:app --host 0.0.0.0 --port 8000

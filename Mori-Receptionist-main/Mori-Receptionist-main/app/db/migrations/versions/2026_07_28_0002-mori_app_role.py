"""add mori_app role without BYPASSRLS so RLS actually applies

The default Neon role (`neondb_owner`) ships with `rolbypassrls = true`, so
Postgres skips RLS on it entirely — even with FORCE ROW LEVEL SECURITY on the
table. Confirmed against the running DB:

    SELECT rolname, rolsuper, rolbypassrls FROM pg_roles WHERE rolname = current_user;
    -- ('neondb_owner', False, True)

That's why `tests/proof_of_rls.py` returned 0/4 while `tests/proof_of_retriever.py`
returned 10/10 — retrieval is safe via the explicit `WHERE tenant_id` in
`app/retrieval/retriever.py`, but the RLS backstop for forgotten filters is
inert.

This migration creates `mori_app` — a LOGIN role WITHOUT the BYPASSRLS
attribute — and grants it the CRUD privileges the app needs on the public
schema. The app then connects as `mori_app` instead of `neondb_owner`, and
FORCE ROW LEVEL SECURITY on `knowledge` starts biting like it was supposed to.

The password is set out of band (Neon dashboard or a one-line SQL by hand). It
does not belong in a migration file — migrations are checked into git.

Migration itself runs as whatever role Alembic is configured with — typically
still `neondb_owner`, which owns the DB and can CREATE ROLE / GRANT freely.
`neondb_owner` continues to exist; `mori_app` is a second, more restricted
identity that the app uses at runtime.

Idempotent: CREATE ROLE errors if the role already exists, so we guard with a
DO block. Re-runs of `alembic upgrade` are safe.

Revision ID: 8a3f1c9e2b47
Revises: 4b6e9c02f7a1
Create Date: 2026-07-28 00:02:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "8a3f1c9e2b47"
down_revision: Union[str, Sequence[str], None] = "4b6e9c02f7a1"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. Create the role, no password (set separately). LOGIN so it can
    #    actually connect. NOBYPASSRLS is the whole point of this migration.
    op.execute("""
        DO $$
        BEGIN
            IF NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mori_app') THEN
                CREATE ROLE mori_app WITH LOGIN NOBYPASSRLS;
            ELSE
                -- Make sure attribute is right even if the role pre-exists.
                ALTER ROLE mori_app WITH LOGIN NOBYPASSRLS;
            END IF;
        END $$;
    """)

    # 2. Grant it read/write on everything in public. USAGE on schema is
    #    required or the app can't see any table by unqualified name.
    op.execute("GRANT USAGE ON SCHEMA public TO mori_app")
    op.execute("GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON ALL TABLES IN SCHEMA public TO mori_app")
    op.execute("GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public TO mori_app")

    # 3. Default privileges so tables/sequences created by FUTURE migrations
    #    (which run as neondb_owner) are automatically visible to mori_app.
    #    Without this, every new table would need a fresh GRANT.
    #    TRUNCATE included so proof_of_rls (and any future ops that reset
    #    tables) don't hit InsufficientPrivilege on tables added later.
    op.execute("""
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT SELECT, INSERT, UPDATE, DELETE, TRUNCATE ON TABLES TO mori_app
    """)
    op.execute("""
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
        GRANT USAGE, SELECT ON SEQUENCES TO mori_app
    """)


def downgrade() -> None:
    # Reverse in reverse order. Default privileges must be revoked before the
    # role can be dropped; existing grants are cleaned up by REASSIGN/DROP.
    op.execute("""
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
        REVOKE USAGE, SELECT ON SEQUENCES FROM mori_app
    """)
    op.execute("""
        ALTER DEFAULT PRIVILEGES IN SCHEMA public
        REVOKE SELECT, INSERT, UPDATE, DELETE ON TABLES FROM mori_app
    """)
    op.execute("REVOKE USAGE, SELECT ON ALL SEQUENCES IN SCHEMA public FROM mori_app")
    op.execute("REVOKE SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public FROM mori_app")
    op.execute("REVOKE USAGE ON SCHEMA public FROM mori_app")
    # DROP OWNED cleans up any remaining privileges Postgres tracks against the
    # role; DROP ROLE then removes the role itself. Wrapped in DO so downgrade
    # is a no-op on a DB where mori_app doesn't exist.
    op.execute("""
        DO $$
        BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mori_app') THEN
                EXECUTE 'DROP OWNED BY mori_app';
                EXECUTE 'DROP ROLE mori_app';
            END IF;
        END $$;
    """)

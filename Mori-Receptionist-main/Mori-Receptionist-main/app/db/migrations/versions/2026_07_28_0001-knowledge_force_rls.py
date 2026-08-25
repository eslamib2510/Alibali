"""force row-level security on knowledge so the table owner is not exempt

The 2026_07_25 migration enabled RLS and created the `tenant_isolation`
policy, but Postgres exempts a table's owner from RLS unless the table is
explicitly set to FORCE. Neon hands the application the owner role, so the
policy was never actually consulted — cross-tenant reads would have gone
through untouched.

With FORCE, the policy applies to everyone including the owner. Sessions must
therefore set the GUC the policy reads:

    SELECT set_config('app.tenant_id', '<uuid>', true)

which `app/db/session.py` does via `get_session(tenant_id=...)`. A session
that doesn't set it sees zero rows on `knowledge` — failing closed, which is
the direction you want for tenant isolation.

Note this also applies to writes: the policy has no separate WITH CHECK, so
Postgres reuses the USING expression for INSERT, and inserting a row whose
tenant_id doesn't match the session GUC is rejected.

Migrations themselves are unaffected — RLS governs DML, not DDL.

Revision ID: 4b6e9c02f7a1
Revises: 7f2b4c8d1a95
Create Date: 2026-07-28 00:01:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "4b6e9c02f7a1"
down_revision: Union[str, Sequence[str], None] = "7f2b4c8d1a95"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.execute("ALTER TABLE knowledge FORCE ROW LEVEL SECURITY")


def downgrade() -> None:
    op.execute("ALTER TABLE knowledge NO FORCE ROW LEVEL SECURITY")

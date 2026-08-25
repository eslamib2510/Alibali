"""tenant webhook_token

Per-tenant webhook auth token. Replaces the global RECEPTIONIST_WEBHOOK_SECRET
shared across all tenants — a leaked tenant token now only exposes that
tenant. The webhook handler looks up the tenant by this value, so it serves
as both authentication and identification.

Nullable here for migration safety on any pre-existing rows; manage_tenant.py
backfills via secrets.token_urlsafe(32) on next CREATE/UPDATE.

Revision ID: 5f3a2e1b9c47
Revises: 9bef0cf00d76
Create Date: 2026-06-09 00:01:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "5f3a2e1b9c47"
down_revision: Union[str, Sequence[str], None] = "9bef0cf00d76"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("webhook_token", sa.String(length=64), nullable=True),
    )
    op.create_index(
        "ix_tenants_webhook_token", "tenants", ["webhook_token"], unique=True
    )


def downgrade() -> None:
    op.drop_index("ix_tenants_webhook_token", table_name="tenants")
    op.drop_column("tenants", "webhook_token")

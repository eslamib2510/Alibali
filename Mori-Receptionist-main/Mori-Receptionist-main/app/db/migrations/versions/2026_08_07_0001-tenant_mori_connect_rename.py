"""rename tenant.chatwoot_* columns to tenant.mori_connect_*

Part of the operator-side rebrand from Chatwoot to Mori-Connect (our
Chatwoot fork). Only the Tenant-level columns are touched: those describe
account/token state on the inbox platform as a whole. The chatwoot_*_id
columns on customer/conversation/message tables are intentionally kept
because they mirror IDs assigned by the platform API, and the name
documents that provenance.

Downgrade reverses the four renames identically. No data movement, only
column name changes, so the migration is instantaneous.

Revision ID: b2f4e7a1d9c8
Revises: 8a3f1c9e2b47
Create Date: 2026-08-07 00:01:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "b2f4e7a1d9c8"
down_revision: Union[str, Sequence[str], None] = "8a3f1c9e2b47"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


RENAMES = [
    ("chatwoot_account_id", "mori_connect_account_id"),
    ("chatwoot_api_token_enc", "mori_connect_api_token_enc"),
    ("chatwoot_agent_bot_id", "mori_connect_agent_bot_id"),
    ("chatwoot_bot_token_enc", "mori_connect_bot_token_enc"),
]


def upgrade() -> None:
    for old, new in RENAMES:
        op.alter_column("tenants", old, new_column_name=new)


def downgrade() -> None:
    for old, new in RENAMES:
        op.alter_column("tenants", new, new_column_name=old)

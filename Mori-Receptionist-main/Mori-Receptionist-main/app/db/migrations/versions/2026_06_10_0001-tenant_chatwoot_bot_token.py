"""tenant chatwoot_bot_token_enc

Add the bot's own api_access_token (captured from Chatwoot's
create_agent_bot response) so we can post replies AS the bot rather than
AS a Chatwoot user. Without this, every bot reply echoes back through the
webhook with sender.type='user', tripping our human-takeover detector and
silencing the bot on its own conversation.

Encrypted at rest, nullable for migration safety — manage_tenant.py
populates on CREATE.

Revision ID: 2c9a1d4e7f53
Revises: 8a1c4d6f2b93
Create Date: 2026-06-10 00:01:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


revision: str = "2c9a1d4e7f53"
down_revision: Union[str, Sequence[str], None] = "8a1c4d6f2b93"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column("chatwoot_bot_token_enc", sa.Text(), nullable=True),
    )


def downgrade() -> None:
    op.drop_column("tenants", "chatwoot_bot_token_enc")

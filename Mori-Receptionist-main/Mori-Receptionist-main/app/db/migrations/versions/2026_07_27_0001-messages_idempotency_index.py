"""unique index on (tenant_id, chatwoot_message_id) for webhook idempotency

Chatwoot delivers agent-bot webhooks at-least-once — `Webhooks::Trigger`
retries on 429/500 — and ARQ retries jobs that raise. Without a uniqueness
guarantee, a redelivered `message_created` re-runs the whole agent flow:
second Gemini call, second reply posted to the customer.

`app/core/agent.py` checks `repository.message_already_handled()` before doing
any work. This index is the backstop for the race where two deliveries pass
that check concurrently — the loser fails on insert in Phase 1, before the LLM
call, so it can never produce a duplicate reply.

Assistant-authored rows have a NULL `chatwoot_message_id`. Postgres permits
unlimited NULLs in a unique index, so they are unaffected.

NOTE: this migration deletes pre-existing duplicate rows (keeping the oldest
of each group) because the index cannot be built while they exist. Those rows
are artifacts of the bug this migration closes, not real data.

Revision ID: 7f2b4c8d1a95
Revises: 3e5b8a2c7d10
Create Date: 2026-07-27 00:01:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op


revision: str = "7f2b4c8d1a95"
down_revision: Union[str, Sequence[str], None] = "3e5b8a2c7d10"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # Drop duplicates first — keep the earliest row per (tenant, message).
    op.execute("""
        DELETE FROM messages m
        USING messages older
        WHERE m.chatwoot_message_id IS NOT NULL
          AND m.tenant_id = older.tenant_id
          AND m.chatwoot_message_id = older.chatwoot_message_id
          AND (older.created_at, older.id) < (m.created_at, m.id)
    """)

    op.execute("""
        CREATE UNIQUE INDEX uq_messages_tenant_chatwoot_msg
            ON messages (tenant_id, chatwoot_message_id)
    """)


def downgrade() -> None:
    op.execute("DROP INDEX IF EXISTS uq_messages_tenant_chatwoot_msg")

"""Message — one turn in a conversation (user, assistant, tool, or system).

We mirror messages from Chatwoot so we can:
  - Reconstruct the rolling context window for the next LLM call cheaply
  - Track cost (tokens in/out, USD) per tenant and per customer
  - Audit what the bot replied with and why

`tool_call_json` is populated when an assistant turn called a tool; that
turn's text content may be empty. The corresponding tool *result* lands as
a separate message with role='tool'.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, Numeric, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow

if TYPE_CHECKING:
    from app.db.models.conversation import Conversation


class Message(Base):
    __tablename__ = "messages"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    conversation_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        nullable=False,
    )

    role: Mapped[str] = mapped_column(String(16), nullable=False)  # user | assistant | tool | system
    content: Mapped[Optional[str]] = mapped_column(Text)

    # When the assistant turn called a tool, the tool-call payload lands here.
    tool_call_json: Mapped[Optional[dict]] = mapped_column(JSONB)
    # When the message *is* a tool result, this names which tool produced it.
    tool_name: Mapped[Optional[str]] = mapped_column(String(64))

    # Cost / observability — populated for assistant turns; null on user/tool.
    tokens_in: Mapped[Optional[int]] = mapped_column(Integer)
    tokens_out: Mapped[Optional[int]] = mapped_column(Integer)
    cost_usd: Mapped[Optional[float]] = mapped_column(Numeric(10, 6))
    latency_ms: Mapped[Optional[int]] = mapped_column(Integer)

    # Mirror back to Chatwoot's message ID where applicable.
    chatwoot_message_id: Mapped[Optional[int]] = mapped_column(BigInteger)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )

    conversation: Mapped["Conversation"] = relationship(
        back_populates="messages", lazy="raise"
    )

    __table_args__ = (
        # The hottest query: "load last N messages for this conversation."
        Index("ix_messages_conv_time", "conversation_id", "created_at"),
        Index("ix_messages_tenant_time", "tenant_id", "created_at"),
        # Idempotency guard. Chatwoot redelivers agent-bot webhooks on 429/500
        # and ARQ retries failed jobs, so the same `message_created` event can
        # arrive more than once. The agent checks for an existing row before
        # doing any work; this index is the backstop for the race where two
        # deliveries pass that check concurrently — the loser hits an
        # IntegrityError in Phase 1, before the LLM call, so it can never
        # produce a second reply.
        #
        # Assistant rows carry NULL here and Postgres allows unlimited NULLs
        # in a unique index, so they're unaffected.
        Index(
            "uq_messages_tenant_chatwoot_msg",
            "tenant_id",
            "chatwoot_message_id",
            unique=True,
        ),
    )

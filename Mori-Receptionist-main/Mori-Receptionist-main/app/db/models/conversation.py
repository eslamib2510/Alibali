"""Conversation — one chat thread.

The full message log lives in Chatwoot (source of truth). We mirror enough
metadata here to (a) drive the agent's decision loop (handoff threshold,
escalation state) and (b) report on per-tenant activity without round-
tripping to Chatwoot for every count.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, DateTime, ForeignKey, Index, Integer, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow

if TYPE_CHECKING:
    from app.db.models.customer import Customer
    from app.db.models.message import Message
    from app.db.models.tenant import Tenant


class Conversation(Base):
    __tablename__ = "conversations"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )
    customer_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("customers.id", ondelete="CASCADE"),
        nullable=False,
    )

    chatwoot_conversation_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    channel: Mapped[str] = mapped_column(String(32), nullable=False)
    status: Mapped[str] = mapped_column(String(16), default="open")  # open | resolved | escalated

    # Handoff / observability counters. `bot_turn_count` is kept for analytics
    # and dashboards; it does NOT trigger forced escalation. The bot replies
    # without limit until it voluntarily escalates (frustration, sensitive
    # topic, explicit request for a human) or the admin takes over in Chatwoot.
    bot_turn_count: Mapped[int] = mapped_column(Integer, default=0, nullable=False)
    last_bot_message_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    escalated_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))
    escalation_reason: Mapped[Optional[str]] = mapped_column(Text)

    started_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    last_message_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    tenant: Mapped["Tenant"] = relationship(back_populates="conversations", lazy="raise")
    customer: Mapped["Customer"] = relationship(back_populates="conversations", lazy="raise")
    messages: Mapped[list["Message"]] = relationship(
        back_populates="conversation",
        lazy="raise",
        order_by="Message.created_at",
    )

    __table_args__ = (
        UniqueConstraint(
            "tenant_id", "chatwoot_conversation_id", name="uq_conv_tenant_chatwoot"
        ),
        Index("ix_conversations_tenant", "tenant_id"),
        Index("ix_conversations_customer", "customer_id"),
        Index("ix_conversations_status", "tenant_id", "status"),
    )

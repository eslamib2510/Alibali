"""Customer — a unified end-user profile per (tenant, Chatwoot contact).

Bridges Chatwoot's contact and (optionally) Medusa's customer into one
record that the bot reasons over. Long-term memory (preferences, past
purchases) hangs off this row — though customer_facts as a separate table
is deferred until v2.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import ARRAY, BigInteger, DateTime, ForeignKey, Index, String, Text, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow

if TYPE_CHECKING:
    from app.db.models.conversation import Conversation
    from app.db.models.tenant import Tenant


class Customer(Base):
    __tablename__ = "customers"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )

    # Identity bridges. Chatwoot is required (we only know about a customer
    # because they showed up in Chatwoot). Medusa is optional — not every
    # contact buys.
    chatwoot_contact_id: Mapped[int] = mapped_column(BigInteger, nullable=False)
    medusa_customer_id: Mapped[Optional[str]] = mapped_column(Text)

    email: Mapped[Optional[str]] = mapped_column(Text)
    phone: Mapped[Optional[str]] = mapped_column(Text)
    name: Mapped[Optional[str]] = mapped_column(Text)
    locale: Mapped[Optional[str]] = mapped_column(String(8))

    # Loose lifecycle tag — refined by the bot over time.
    lifecycle_stage: Mapped[str] = mapped_column(String(16), default="lead")
    tags: Mapped[list[str]] = mapped_column(ARRAY(Text), default=list)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    last_seen_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    tenant: Mapped["Tenant"] = relationship(back_populates="customers", lazy="raise")
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="customer", lazy="raise"
    )

    __table_args__ = (
        # One Chatwoot contact maps to at most one Customer per tenant.
        # Webhook spam (same event delivered twice) won't create duplicates.
        UniqueConstraint("tenant_id", "chatwoot_contact_id", name="uq_customer_tenant_chatwoot"),
        Index("ix_customers_tenant", "tenant_id"),
        Index("ix_customers_email", "tenant_id", "email"),
    )

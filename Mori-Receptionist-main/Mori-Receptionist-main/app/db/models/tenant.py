"""Tenant — a client we serve (AlaBali itself, or a white-label customer).

The routing table: a Mori-Connect webhook arrives with `account_id`, we look
up the matching tenant by `mori_connect_account_id`, and load that tenant's
prompt, Medusa creds, LLM choice, etc.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING, Optional

from sqlalchemy import BigInteger, Boolean, DateTime, String, Text
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow

if TYPE_CHECKING:
    from app.db.models.conversation import Conversation
    from app.db.models.customer import Customer


class Tenant(Base):
    __tablename__ = "tenants"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )

    slug: Mapped[str] = mapped_column(String(64), unique=True, nullable=False)
    name: Mapped[str] = mapped_column(String(128), nullable=False)

    # Routing: one inbox platform account per tenant.
    mori_connect_account_id: Mapped[int] = mapped_column(
        BigInteger, unique=True, nullable=False
    )
    # Token to make API calls (post messages, create custom attributes,
    # toggle status) on this tenant's account. Encrypted at rest.
    mori_connect_api_token_enc: Mapped[Optional[str]] = mapped_column(Text)
    # Agent Bot record we created in this tenant's inbox platform account.
    # Populated by scripts/manage_tenant.py on CREATE so we don't try to
    # re-create it on subsequent invocations. Stays null for chat-only tenants
    # that don't have a token yet.
    mori_connect_agent_bot_id: Mapped[Optional[int]] = mapped_column(BigInteger)
    # The bot's OWN api_access_token, captured from the platform's
    # create_agent_bot response. Used for posting replies so outgoing messages
    # come back with sender.type='agent_bot' instead of 'user'; keeps our
    # human-takeover detector from misfiring on the bot's own echo webhooks.
    # Encrypted at rest.
    mori_connect_bot_token_enc: Mapped[Optional[str]] = mapped_column(Text)

    # Per-tenant webhook auth token. Generated on CREATE, embedded in the
    # bot's outgoing_url as `?token=<this>`. The webhook handler looks up
    # the tenant by this value — so it's BOTH authentication AND tenant
    # identification in one step. Replaces a single shared secret across
    # all tenants, which any tenant's admins could see and use to spoof
    # webhooks for other tenants.
    #
    # Nullable for migration safety on pre-existing rows; manage_tenant.py
    # generates one on every CREATE so new rows always have one.
    webhook_token: Mapped[Optional[str]] = mapped_column(
        String(64), unique=True, index=True
    )

    # Commerce — optional. Chat-only tenants have these null.
    medusa_api_url: Mapped[Optional[str]] = mapped_column(Text)
    medusa_api_key_enc: Mapped[Optional[str]] = mapped_column(Text)

    # Agent config. Model + provider are NOT per-tenant — they're operator-wide
    # via settings.RECEPTIONIST_DEFAULT_MODEL. If you ever need per-tenant
    # overrides (premium tier, A/B, etc.), add a nullable `llm_model_override`
    # column and let it fall back to the env default.
    prompt_template: Mapped[str] = mapped_column(Text, nullable=False)

    # Operating mode. The bot always replies and there's no forced count-based
    # handoff — admin presence is signalled instead via private notes.
    #   True  → post a Chatwoot private note on each incoming customer message,
    #           so admin is aware and can intervene at will. ("Notify Always")
    #   False → operate silently; admin can still see / take over any conversation
    #           via Chatwoot UI, but no per-message notification is posted. ("Silent")
    notify_admin_on_message: Mapped[bool] = mapped_column(
        Boolean, default=True, nullable=False
    )

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow
    )
    deleted_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True))

    customers: Mapped[list["Customer"]] = relationship(
        back_populates="tenant", lazy="raise"
    )
    conversations: Mapped[list["Conversation"]] = relationship(
        back_populates="tenant", lazy="raise"
    )

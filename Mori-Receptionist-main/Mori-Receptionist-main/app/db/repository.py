"""Database access layer for the agent loop.

All DB reads/writes go through here so the agent code stays free of SQLAlchemy
specifics. Easier to test, easier to swap engines if we ever need to.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models.conversation import Conversation
from app.db.models.customer import Customer
from app.db.models.message import Message
from app.db.models.tenant import Tenant


# ─── Tenants ─────────────────────────────────────────────────────────────────


def get_tenant_by_mori_connect_account(db: Session, account_id: int) -> Optional[Tenant]:
    return db.scalar(
        select(Tenant).where(Tenant.mori_connect_account_id == account_id)
    )


def get_tenant_by_webhook_token(db: Session, webhook_token: str) -> Optional[Tenant]:
    """Resolve a tenant from the webhook auth token in the URL query string.
    This is the auth principle for incoming Mori-Connect webhooks: a tenant
    IS whoever's token validated the request."""
    if not webhook_token:
        return None
    return db.scalar(
        select(Tenant).where(Tenant.webhook_token == webhook_token)
    )


def get_tenant_by_id(db: Session, tenant_id) -> Optional[Tenant]:
    """Re-fetch a tenant inside a fresh session. Used by the agent loop after
    the webhook route has already authenticated by token and passed the id."""
    return db.scalar(select(Tenant).where(Tenant.id == tenant_id))


# ─── Customers ───────────────────────────────────────────────────────────────


def upsert_customer(
    db: Session,
    *,
    tenant: Tenant,
    chatwoot_contact_id: int,
    name: Optional[str] = None,
    email: Optional[str] = None,
    phone: Optional[str] = None,
) -> Customer:
    """Find or create a Customer for this (tenant, chatwoot_contact_id) pair.

    Mirrors basic profile fields if Chatwoot has fresher info. Keeps the row
    idempotent — webhook deliveries are at-least-once."""
    customer = db.scalar(
        select(Customer).where(
            Customer.tenant_id == tenant.id,
            Customer.chatwoot_contact_id == chatwoot_contact_id,
        )
    )
    if customer is None:
        customer = Customer(
            tenant_id=tenant.id,
            chatwoot_contact_id=chatwoot_contact_id,
            name=name,
            email=email,
            phone=phone,
            last_seen_at=utcnow(),
        )
        db.add(customer)
        db.flush()  # populate customer.id
        return customer

    # Light merge: only fill blanks; don't overwrite values the agent enriched.
    if not customer.name and name:
        customer.name = name
    if not customer.email and email:
        customer.email = email
    if not customer.phone and phone:
        customer.phone = phone
    customer.last_seen_at = utcnow()
    return customer


# ─── Conversations ───────────────────────────────────────────────────────────


def get_conversation_by_pk(db: Session, conversation_pk) -> Optional[Conversation]:
    """Fetch a conversation by our internal primary key. Used by the agent's
    post-reply phase to refetch the row in a fresh session for the assistant
    message save + escalation state update."""
    return db.scalar(select(Conversation).where(Conversation.id == conversation_pk))


def upsert_conversation(
    db: Session,
    *,
    tenant: Tenant,
    customer: Customer,
    chatwoot_conversation_id: int,
    channel: str,
) -> Conversation:
    """Find or create the Conversation row for a Chatwoot conversation."""
    conv = db.scalar(
        select(Conversation).where(
            Conversation.tenant_id == tenant.id,
            Conversation.chatwoot_conversation_id == chatwoot_conversation_id,
        )
    )
    if conv is None:
        conv = Conversation(
            tenant_id=tenant.id,
            customer_id=customer.id,
            chatwoot_conversation_id=chatwoot_conversation_id,
            channel=channel,
            started_at=utcnow(),
            last_message_at=utcnow(),
        )
        db.add(conv)
        db.flush()
        return conv

    conv.last_message_at = utcnow()
    return conv


# ─── Messages ────────────────────────────────────────────────────────────────


def message_already_handled(
    db: Session, *, tenant_id, chatwoot_message_id: Optional[int]
) -> bool:
    """True if we've already stored this Chatwoot message for this tenant.

    Webhook deliveries are at-least-once: Chatwoot retries agent-bot webhooks
    that answer 429/500, and ARQ retries jobs that raise. Without this check a
    redelivery re-runs the whole flow — second LLM call, second reply to the
    customer. Callers should bail out early when this returns True.
    """
    if chatwoot_message_id is None:
        return False
    return db.scalar(
        select(Message.id)
        .where(
            Message.tenant_id == tenant_id,
            Message.chatwoot_message_id == chatwoot_message_id,
        )
        .limit(1)
    ) is not None


def save_message(
    db: Session,
    *,
    tenant: Tenant,
    conversation: Conversation,
    role: str,
    content: Optional[str],
    chatwoot_message_id: Optional[int] = None,
    tool_call_json: Optional[dict] = None,
    tool_name: Optional[str] = None,
    tokens_in: Optional[int] = None,
    tokens_out: Optional[int] = None,
    cost_usd: Optional[float] = None,
    latency_ms: Optional[int] = None,
) -> Message:
    msg = Message(
        tenant_id=tenant.id,
        conversation_id=conversation.id,
        role=role,
        content=content,
        chatwoot_message_id=chatwoot_message_id,
        tool_call_json=tool_call_json,
        tool_name=tool_name,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost_usd=cost_usd,
        latency_ms=latency_ms,
    )
    db.add(msg)
    db.flush()
    return msg


def recent_messages(
    db: Session, conversation: Conversation, limit: int = 10
) -> list[tuple[str, str]]:
    """Return up to `limit` most recent (role, content) tuples, oldest first.
    Skips messages with empty content (tool calls without text)."""
    rows = (
        db.execute(
            select(Message.role, Message.content)
            .where(Message.conversation_id == conversation.id)
            .order_by(Message.created_at.desc())
            .limit(limit)
        )
        .tuples()
        .all()
    )
    # rows came in newest-first; flip and drop empties.
    return [(role, content) for role, content in reversed(rows) if content]

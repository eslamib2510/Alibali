"""Mori-Connect client + webhook payload parsing.

Two responsibilities:
  1. Parse the inbound Agent Bot webhook into a typed event we can switch on.
  2. Talk back to the platform's REST API (post a reply, eventually set status).

Mori-Connect is our Chatwoot fork; the wire format (endpoints, JSON shapes)
is identical, so the docs at https://www.chatwoot.com/developers/api still
apply verbatim.
"""

from __future__ import annotations

from typing import Any, Optional

import httpx
from pydantic import BaseModel, Field

from app.config import settings


# ─── Webhook payload models ──────────────────────────────────────────────────


class _Account(BaseModel):
    id: int


class _Conversation(BaseModel):
    id: int
    status: str  # 'open' | 'pending' | 'resolved' | 'snoozed'
    # Per-conversation overrides set by admin in the platform UI. We read
    # `bot_mode` here to decide whether to post an admin-notification private
    # note. Defaults are tenant-level (see Tenant.notify_admin_on_message).
    custom_attributes: dict[str, Any] = Field(default_factory=dict)


class _Sender(BaseModel):
    id: int
    name: Optional[str] = None
    email: Optional[str] = None
    phone_number: Optional[str] = None
    type: Optional[str] = None  # 'contact' (customer) | 'user' (human agent) | 'agent_bot' (us)


class _Inbox(BaseModel):
    id: int
    channel_type: Optional[str] = None  # 'Channel::WebWidget' | 'Channel::Whatsapp' | ...


class MoriConnectWebhookEvent(BaseModel):
    """Just the fields v0 needs. Extra fields are ignored by pydantic."""

    event: str = Field(..., description="e.g. 'message_created'")
    id: Optional[int] = None  # the message id, when event=message_created
    content: Optional[str] = None
    message_type: Optional[str] = None  # 'incoming' = from customer, 'outgoing' = from bot/agent
    account: _Account
    conversation: _Conversation
    sender: Optional[_Sender] = None
    inbox: Optional[_Inbox] = None

    @property
    def is_incoming_customer_message(self) -> bool:
        """True when this is a customer message we should consider replying to.
        Outgoing messages from us or other agents come through the same webhook
        and we must ignore them or we'll echo forever."""
        return self.event == "message_created" and self.message_type == "incoming"

    @property
    def is_human_agent_takeover(self) -> bool:
        """True when an outgoing message was sent by a human agent (not our
        own bot). This is the handoff signal we use to mark the conversation
        escalated locally — we don't trust Chatwoot's `status` field for this
        because of the v4.7.0 regression (see core/agent.py)."""
        return (
            self.event == "message_created"
            and self.message_type == "outgoing"
            and self.sender is not None
            and (self.sender.type or "").lower() == "user"
        )


# ─── Mori-Connect API client ─────────────────────────────────────────────────


class MoriConnectClient:
    """Thin wrapper around the platform's account-scoped REST API."""

    def __init__(self, api_token: str, base_url: Optional[str] = None):
        self.base_url = (base_url or settings.MORI_CONNECT_BASE_URL).rstrip("/")
        self.api_token = api_token

    def _headers(self) -> dict[str, str]:
        return {
            "Content-Type": "application/json",
            "api_access_token": self.api_token,
        }

    async def post_message(
        self,
        account_id: int,
        conversation_id: int,
        content: str,
        private: bool = False,
    ) -> dict[str, Any]:
        """Send a message into a conversation. `private=True` posts a private
        note visible only to agents (used for handoff context)."""
        url = (
            f"{self.base_url}/api/v1/accounts/{account_id}"
            f"/conversations/{conversation_id}/messages"
        )
        body = {
            "content": content,
            "message_type": "outgoing",
            "private": private,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, json=body, headers=self._headers())
            r.raise_for_status()
            return r.json()

    async def toggle_status(
        self, account_id: int, conversation_id: int, status: str
    ) -> dict[str, Any]:
        """Set a conversation's status. `status` in ('open','resolved','pending','snoozed')."""
        url = (
            f"{self.base_url}/api/v1/accounts/{account_id}"
            f"/conversations/{conversation_id}/toggle_status"
        )
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, json={"status": status}, headers=self._headers())
            r.raise_for_status()
            return r.json()

    async def ensure_custom_attribute(
        self,
        account_id: int,
        attribute_definition: dict[str, Any],
    ) -> Optional[dict[str, Any]]:
        """Idempotently create a custom attribute definition in a Chatwoot
        account. Returns the new definition, or None if it already exists.

        Used during tenant onboarding so admins never have to create the
        `bot_mode` attribute (or any future attributes we depend on) by hand.
        """
        url = (
            f"{self.base_url}/api/v1/accounts/{account_id}"
            f"/custom_attribute_definitions"
        )
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(
                url, json=attribute_definition, headers=self._headers()
            )
            # 422 commonly means a duplicate attribute_key; treat as no-op.
            if r.status_code == 422:
                return None
            r.raise_for_status()
            return r.json()

    async def ensure_bot_mode_attribute(
        self, account_id: int
    ) -> Optional[dict[str, Any]]:
        """Convenience wrapper: provision the `bot_mode` custom attribute that
        the agent reads per-conversation to decide whether to notify admin.
        Call once during tenant onboarding."""
        return await self.ensure_custom_attribute(account_id, BOT_MODE_ATTRIBUTE)

    async def create_agent_bot(
        self,
        *,
        account_id: int,
        name: str,
        outgoing_url: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Create an Agent Bot in this tenant's Chatwoot account. Returns the
        new bot record (including its `id`).

        Endpoint is account-scoped (`/api/v1/accounts/{account_id}/agent_bots`)
        — the platform-scoped `/api/v1/agent_bots` is super-admin only and
        404s for normal users on Chatwoot Cloud.

        Used by `scripts/manage_tenant.py` so onboarding a tenant is one
        command end-to-end, no UI clicking in Chatwoot."""
        url = f"{self.base_url}/api/v1/accounts/{account_id}/agent_bots"
        body = {
            "name": name,
            "description": description,
            "outgoing_url": outgoing_url,
        }
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, json=body, headers=self._headers())
            r.raise_for_status()
            return r.json()

    async def list_inboxes(self, account_id: int) -> list[dict[str, Any]]:
        """List all inboxes in this account. Used by tenant onboarding to
        decide whether to auto-attach the new bot."""
        url = f"{self.base_url}/api/v1/accounts/{account_id}/inboxes"
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.get(url, headers=self._headers())
            r.raise_for_status()
            data = r.json()
        # Chatwoot returns either {"payload": [...]} or a bare list. Handle both.
        if isinstance(data, dict) and "payload" in data:
            return list(data["payload"])
        return list(data)

    async def set_agent_bot_on_inbox(
        self,
        account_id: int,
        inbox_id: int,
        agent_bot_id: Optional[int],
    ) -> None:
        """Attach (or detach) an Agent Bot to a specific inbox. Pass
        `agent_bot_id=None` to clear the assignment."""
        url = (
            f"{self.base_url}/api/v1/accounts/{account_id}"
            f"/inboxes/{inbox_id}/set_agent_bot"
        )
        body = {"agent_bot": agent_bot_id}
        async with httpx.AsyncClient(timeout=20.0) as client:
            r = await client.post(url, json=body, headers=self._headers())
            r.raise_for_status()


# ─── Provisionable custom attributes ─────────────────────────────────────────


# Per-conversation override for the admin-notification mode. Tenants get this
# attribute provisioned in their Chatwoot account at onboarding (see
# MoriConnectClient.ensure_bot_mode_attribute). Admin then sets it on individual
# conversations from the Chatwoot UI as a dropdown.
BOT_MODE_ATTRIBUTE: dict[str, Any] = {
    "attribute_display_name": "Bot Mode",
    "attribute_key": "bot_mode",
    # Chatwoot type codes: 0 text, 1 number, 2 currency, 3 percent, 4 link,
    # 5 date, 6 list, 7 checkbox.
    "attribute_display_type": 6,
    "attribute_values": ["silent", "notify_always"],
    # 0 = conversation-scoped, 1 = contact-scoped.
    "attribute_model": 0,
    "attribute_description": (
        "Override admin-notification mode for this conversation. "
        "`silent` = bot replies without notifying admin. "
        "`notify_always` = bot also posts a private note for admin on each "
        "incoming customer message. If left blank, the tenant default applies."
    ),
}

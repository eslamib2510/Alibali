"""The receptionist's main loop.

Single entry point: `handle_message_event(payload, tenant_id=...)`.

The tenant_id is trusted — the webhook route resolves it from the per-tenant
`webhook_token` query param, so by the time we get here the request has been
authenticated AND attributed to a specific tenant.

What it does, in order:
  1. Skip non-incoming events (we'd echo our own messages otherwise).
  2. Re-fetch the tenant inside our own session; verify the payload's
     account_id matches (defense against a leaked token being used to spoof
     webhooks for the wrong account).
  3. Respect Chatwoot's conversation status — only auto-reply when 'pending'.
     A human has taken over once it's 'open'.
  4. Upsert customer + conversation rows.
  5. Save the incoming message.
  5a. Voluntary escalation: if the customer is asking for a human, flip
      status to 'open' in Chatwoot and stop.
  6. Build (system_prompt, history) and ask Gemini for a reply.
  7. Save the reply, then POST it back to Chatwoot.
  7a. Admin notification: if this tenant/conversation is in "notify_always"
      mode, also post a private note so the admin sees what just happened.

This file is *just orchestration*. Provider-specific stuff lives in
integrations/* and tools/* will hang off the LLM call in a later phase.
"""

from __future__ import annotations

import logging
import time

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select

from app.agent.graph import build_graph
from app.core import crypto
from app.db import repository
from app.db.base import utcnow
from app.db.models.conversation import Conversation
from app.db.models.tenant import Tenant
from app.db.session import get_session
from app.integrations.mori_connect import MoriConnectClient, MoriConnectWebhookEvent

logger = logging.getLogger(__name__)


# Escalation is now handled INSIDE the graph:
#   - ESCALATION_INSTRUCTIONS lives in app.agent.graph and is auto-appended
#     to the tenant prompt every turn
#   - The check_escalation node fills state["escalation_reason"] if the LLM
#     outputs the ESCALATE: marker
# Phase 3 below reads that state field directly, so no local text parsing.


def _history_to_messages(history: list[tuple[str, str]]) -> list:
    """Convert repository history tuples into LangChain messages the graph
    accepts. Skips 'tool' rows — the graph regenerates those inline each turn
    from the tool it just called, and replaying old tool rows without their
    original tool_call ids would fail LangChain's validation."""
    messages: list = []
    for role, content in history:
        if role == "user":
            messages.append(HumanMessage(content=content))
        elif role == "assistant":
            messages.append(AIMessage(content=content))
        # 'tool' / 'system' rows are dropped intentionally.
    return messages


async def handle_message_event(payload: dict, *, tenant_id) -> None:
    """Process one Chatwoot webhook delivery. Swallows known no-ops; logs and
    re-raises anything unexpected so the route can return 500 and Chatwoot
    will retry.

    `tenant_id` is the authenticated tenant id, resolved by the webhook route
    from the per-tenant `webhook_token`. It is trusted; payload contents are
    not (we cross-check mori_connect_account_id below)."""
    # Chatwoot's Agent Bot webhook fires several event types — message_created,
    # conversation_created, conversation_updated, contact_created, etc. They
    # all hit this URL but have different payload shapes. Filter on `event`
    # BEFORE validating: only message_created has the shape our model expects.
    if payload.get("event") != "message_created":
        return

    event = MoriConnectWebhookEvent.model_validate(payload)

    # 1. Only respond to customer messages. Outgoing (us / agents) bounces back
    # through the same webhook — replying to those would loop forever.
    if not event.is_incoming_customer_message:
        # Before bailing: if this was a HUMAN agent's reply (not our own bot),
        # mark the conversation escalated so we stay silent on future customer
        # messages. Without this, the bot would happily keep replying alongside
        # the admin who just took over.
        if event.is_human_agent_takeover:
            with get_session() as db:
                conversation = db.scalar(
                    select(Conversation).where(
                        Conversation.tenant_id == tenant_id,
                        Conversation.chatwoot_conversation_id == event.conversation.id,
                    )
                )
                if conversation is not None and conversation.status != "escalated":
                    conversation.status = "escalated"
                    conversation.escalated_at = utcnow()
                    conversation.escalation_reason = "human agent replied"
                    logger.info(
                        "Conv %s: human agent took over, bot silenced",
                        event.conversation.id,
                    )
        return

    # 3. Bot-silence check moved INTO Phase 1: we no longer trust Chatwoot's
    # `conversation.status` ("open" vs "pending") because Chatwoot v4.7.0+ on
    # Cloud has a regression that flips conversations to `open` even when
    # the bot is working correctly (see GH chatwoot/chatwoot#12754). That
    # would silence the bot after the first reply for every customer.
    # Instead we read OUR own `conversations.status` ("escalated") which we
    # only set when the bot voluntarily hands off. Admin override is via the
    # Chatwoot custom attribute `bot_mode = silent`.

    # ─── Phase 1: critical setup (DB read + Chatwoot client) ────────────────
    with get_session() as db:
        # Re-fetch tenant by trusted id; verify payload's account matches
        # (token spoofing defense).
        tenant = repository.get_tenant_by_id(db, tenant_id)
        if tenant is None:
            logger.error("Authenticated tenant_id=%s vanished from DB", tenant_id)
            return
        if event.account.id != tenant.mori_connect_account_id:
            logger.warning(
                "Token/account mismatch: tenant=%s expects account_id=%s, "
                "payload claims %s; refusing to process",
                tenant.slug, tenant.mori_connect_account_id, event.account.id,
            )
            return
        # IDEMPOTENCY GATE. Webhook delivery is at-least-once — Chatwoot
        # redelivers on 429/500 and ARQ retries jobs that raise. Re-running
        # this flow for a message we've already stored means a second Gemini
        # call and a second reply landing in the customer's thread. Check
        # before doing any work; the unique index on
        # (tenant_id, chatwoot_message_id) covers the concurrent-delivery race.
        if repository.message_already_handled(
            db, tenant_id=tenant_id, chatwoot_message_id=event.id
        ):
            logger.info(
                "Chatwoot message %s already handled for tenant %s — "
                "skipping duplicate delivery",
                event.id, tenant.slug,
            )
            return

        # Use the BOT's own access_token (captured at agent-bot creation) so
        # Chatwoot stamps outgoing messages with sender.type='agent_bot'.
        # Without this the bot's own echo webhooks come back as 'user' and
        # trip is_human_agent_takeover, silencing the conversation on every
        # bot reply. Refuse to reply if the bot token is missing — falling
        # back to the user token would re-introduce that exact bug.
        try:
            mori_connect_token = crypto.get_mori_connect_bot_token(tenant)
        except crypto.EncryptionError as e:
            logger.error(
                "Tenant %s — bot token missing, refusing to reply (would "
                "trigger self-echo bug): %s. Re-run manage_tenant.py to "
                "capture the bot's access_token.",
                tenant.slug, e,
            )
            return

        # Stash everything we need outside the session so we don't hold a
        # Postgres connection during the (slow) LLM call.
        tenant_prompt = tenant.prompt_template
        tenant_notify_default = tenant.notify_admin_on_message
        tenant_slug = tenant.slug

        chatwoot = MoriConnectClient(api_token=mori_connect_token)

        sender = event.sender
        customer = repository.upsert_customer(
            db, tenant=tenant,
            chatwoot_contact_id=sender.id if sender else 0,
            name=sender.name if sender else None,
            email=sender.email if sender else None,
            phone=sender.phone_number if sender else None,
        )
        channel = (event.inbox.channel_type if event.inbox else "unknown") or "unknown"
        conversation = repository.upsert_conversation(
            db, tenant=tenant, customer=customer,
            chatwoot_conversation_id=event.conversation.id, channel=channel,
        )

        # ADMIN HAND-BACK detection: if we're escalated locally BUT Chatwoot
        # now says the conversation is `pending` or `resolved`, the admin
        # clicked "Mark as pending" or "Resolve" to hand control back to us.
        # Asymmetric trust: we ignore Chatwoot's `open` (Chatwoot bug flips
        # it wrongly) but `pending`/`resolved` are only set by intentional
        # admin action. Clear escalation and fall through to normal flow.
        if (
            conversation.status == "escalated"
            and event.conversation.status in ("pending", "resolved")
        ):
            conversation.status = "pending"
            conversation.escalated_at = None
            conversation.escalation_reason = None
            logger.info(
                "Conv %s: admin handed control back to bot (chatwoot_status=%s)",
                event.conversation.id, event.conversation.status,
            )
            # don't return — continue normal processing of this customer message

        # OUR OWN silence check — we marked this conversation escalated in a
        # prior turn and admin hasn't handed it back. Save the customer
        # message for history and stop.
        elif conversation.status == "escalated":
            repository.save_message(
                db, tenant=tenant, conversation=conversation,
                role="user", content=event.content,
                chatwoot_message_id=event.id,
            )
            logger.info(
                "Conversation %s escalated locally — bot silent",
                event.conversation.id,
            )
            return

        repository.save_message(
            db, tenant=tenant, conversation=conversation,
            role="user", content=event.content, chatwoot_message_id=event.id,
        )
        history = repository.recent_messages(db, conversation, limit=10)
        conversation_pk = conversation.id  # capture for Phase 4

    # ─── Phase 2: agent graph (outside DB — no connection held) ─────────────
    # The graph replaces the single-shot Gemini call from v1. Each turn it
    # can call search_knowledge (and future tools) as many times as it needs
    # before answering. Native function-calling — no ReAct text parsing.
    #
    # ChatGoogleGenerativeAI.ainvoke is async native (uses aiohttp under the
    # hood), so `asyncio.to_thread` isn't needed here the way it was for the
    # sync `generate_reply`. Concurrent conversations run in parallel on the
    # same event loop without stepping on each other.
    #
    # graph is built per turn: the search_knowledge tool needs tenant_id
    # bound at construction time, and tenant_prompt can change between
    # invocations. Compilation is cheap (no LLM warm-up) so no caching.
    t0 = time.monotonic()
    graph = build_graph(tenant_id=str(tenant_id), tenant_prompt=tenant_prompt)
    initial_state = {
        "tenant_id": str(tenant_id),
        "messages": [
            *_history_to_messages(history),
            HumanMessage(content=event.content or ""),
        ],
        "escalation_reason": None,
    }
    try:
        final_state = await graph.ainvoke(initial_state)
    except Exception:
        logger.exception("Agent graph failed for tenant=%s", tenant_slug)
        raise

    latency_ms = int((time.monotonic() - t0) * 1000)
    last_message = final_state["messages"][-1]
    llm_output = (
        last_message.content
        if isinstance(last_message.content, str)
        else str(last_message.content)
    )
    usage = getattr(last_message, "usage_metadata", None) or {}
    if not llm_output:
        logger.warning("Agent returned empty text for conv %s", event.conversation.id)
        return

    # ─── Phase 3: decide reply, post it ASAP ────────────────────────────────
    # escalation_reason is set by the graph's check_escalation node when the
    # LLM output the ESCALATE: marker. No local parsing needed.
    escalation_reason = final_state.get("escalation_reason")
    if escalation_reason:
        reply_text = (
            "Let me get one of our teammates to help you. "
            "They'll reply here as soon as they're available."
        )
        should_open_status = True
        notify_admin = True  # always tell admin about handoffs
        private_note = (
            f"🤝 Bot escalated. Reason: {escalation_reason}. "
            "Bot is now silent on this thread."
        )
    else:
        reply_text = llm_output
        should_open_status = False
        # Per-conversation bot_mode override beats the tenant default.
        bot_mode_override = (event.conversation.custom_attributes or {}).get("bot_mode")
        if bot_mode_override == "notify_always":
            notify_admin = True
        elif bot_mode_override == "silent":
            notify_admin = False
        else:
            notify_admin = bool(tenant_notify_default)
        private_note = (
            f"🤖 AI replied to: {event.content!r}\n\n"
            f"Bot: {reply_text!r}\n\n"
            "Reply in this thread to take over. Set the `bot_mode` "
            "custom attribute to `silent` to mute these notes."
        ) if notify_admin else None

    try:
        await chatwoot.post_message(
            account_id=event.account.id,
            conversation_id=event.conversation.id,
            content=reply_text,
        )
    except Exception:
        logger.exception("Failed to post reply to Chatwoot for conv %s", event.conversation.id)
        raise

    # ─── Phase 4: persist assistant turn + side effects (post-reply) ────────
    # Customer already got their reply in Phase 3. Everything here is
    # bookkeeping — never raise. Re-raising would 500 the route, Chatwoot
    # would redeliver the webhook, and the LLM + reply would run AGAIN,
    # double-messaging the customer. Log and swallow instead.
    try:
        with get_session() as db:
            tenant_row = repository.get_tenant_by_id(db, tenant_id)
            conversation = repository.get_conversation_by_pk(db, conversation_pk)
            if tenant_row is not None and conversation is not None:
                repository.save_message(
                    db, tenant=tenant_row, conversation=conversation,
                    role="assistant", content=reply_text,
                    # LangChain's ChatModel exposes token counts under
                    # `usage_metadata` with keys "input_tokens" / "output_tokens".
                    # We only capture the last turn's usage — if the graph made
                    # multiple LLM calls (tool loop), earlier turns' tokens are
                    # lost. Fine for cost-tracking; refine later if per-call
                    # accounting matters.
                    tokens_in=usage.get("input_tokens"),
                    tokens_out=usage.get("output_tokens"),
                    latency_ms=latency_ms,
                )
                if escalation_reason:
                    conversation.status = "escalated"
                    conversation.escalated_at = utcnow()
                    conversation.escalation_reason = escalation_reason
                else:
                    conversation.bot_turn_count = (conversation.bot_turn_count or 0) + 1
                    conversation.last_bot_message_at = utcnow()
    except Exception:
        logger.exception(
            "Phase 4 bookkeeping failed after reply was posted "
            "(conversation=%s) — assistant turn may be missing from DB",
            event.conversation.id,
        )

    if private_note:
        try:
            await chatwoot.post_message(
                account_id=event.account.id,
                conversation_id=event.conversation.id,
                content=private_note, private=True,
            )
        except Exception:
            logger.exception("Failed to post admin private note")

    if should_open_status:
        try:
            await chatwoot.toggle_status(
                account_id=event.account.id,
                conversation_id=event.conversation.id,
                status="open",
            )
        except Exception:
            logger.exception("Failed to flip conversation to 'open' on handoff")

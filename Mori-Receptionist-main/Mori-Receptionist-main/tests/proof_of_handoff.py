"""Handoff lifecycle harness.

Run inside the app image with a live Postgres:

    python -m tests.proof_of_handoff

Covers the three transitions that decide whether the bot speaks. Each fails
silently in production if broken — the bot either goes mute for a customer or
talks over a human agent, and nothing raises.

  CASE 1  Voluntary escalation. The model answers `ESCALATE: <reason>`. The
          customer must get the handoff line (not the raw marker), the
          conversation must be marked escalated with the reason stored, and
          Chatwoot's status must be flipped to `open`.

  CASE 2  Bot stays silent afterwards. The next customer message must be
          stored for history but must NOT trigger the LLM or a reply.

  CASE 3  Human agent takeover. An outgoing message from a human agent
          (sender.type == 'user') must silence the bot, even though the bot
          never escalated.

  CASE 4  Admin hand-back. Once escalated, a customer message arriving while
          Chatwoot reports `pending` means the admin handed control back. The
          bot must resume: clear escalation and reply again.

  CASE 5  Own echo is ignored. The bot's own outgoing message
          (sender.type == 'agent_bot') must NOT be read as a takeover — this
          is the self-silencing bug the bot token exists to prevent.

Gemini and Chatwoot are stubbed; everything else is real.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage
from sqlalchemy import func, select, text

from app.core import crypto
from app.db.models.conversation import Conversation
from app.db.models.message import Message
from app.db.models.tenant import Tenant
from app.db.session import get_session
from tests._helpers import cleanup_by_mori_connect_account_id

ACCOUNT_ID = -5150
CONVERSATION_ID = -8080

results: list[tuple[str, bool, str]] = []
_msg_id = iter(range(700001, 700099))


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {detail}")


def seed_tenant() -> uuid.UUID:
    """Create a fresh test tenant, scoped to the negative test ACCOUNT_ID.

    Only cleans up prior leftovers from THIS harness (by ACCOUNT_ID), never
    other tenants. Real inbox platform account ids are positive so the
    negative value cannot collide with a live tenant.
    """
    with get_session() as db:
        cleanup_by_mori_connect_account_id(db, ACCOUNT_ID)
        tenant = Tenant(
            slug=f"handoff-{uuid.uuid4().hex[:8]}",
            name="Handoff Tenant",
            mori_connect_account_id=ACCOUNT_ID,
            prompt_template="You are a helpful receptionist.",
            webhook_token=f"tok_{uuid.uuid4().hex[:16]}",
            mori_connect_bot_token_enc=crypto.encrypt("fake-bot-token"),
            notify_admin_on_message=False,
        )
        db.add(tenant)
        db.flush()
        return tenant.id


def incoming(content: str, chatwoot_status: str = "pending") -> dict:
    """A customer message."""
    return {
        "event": "message_created",
        "id": next(_msg_id),
        "content": content,
        "message_type": "incoming",
        "account": {"id": ACCOUNT_ID},
        "conversation": {"id": CONVERSATION_ID, "status": chatwoot_status,
                         "custom_attributes": {}},
        "sender": {"id": 55, "name": "Test Customer", "type": "contact"},
        "inbox": {"id": 9, "channel_type": "Channel::Whatsapp"},
    }


def outgoing(content: str, sender_type: str) -> dict:
    """An outgoing message echoed back — from a human agent or from our bot."""
    return {
        "event": "message_created",
        "id": next(_msg_id),
        "content": content,
        "message_type": "outgoing",
        "account": {"id": ACCOUNT_ID},
        "conversation": {"id": CONVERSATION_ID, "status": "open",
                         "custom_attributes": {}},
        "sender": {"id": 99, "name": "Human Agent", "type": sender_type},
        "inbox": {"id": 9, "channel_type": "Channel::Whatsapp"},
    }


class Recorder:
    """Captures what the agent tried to send to Chatwoot and to the graph."""

    def __init__(self, reply: str) -> None:
        self.reply = reply
        self.public: list[str] = []
        self.private: list[str] = []
        self.statuses: list[str] = []
        self.llm_calls = 0

    def build_graph_stub(self, *, tenant_id, tenant_prompt):
        """Stand-in for app.agent.graph.build_graph.

        Returns a minimal compiled-graph shape whose `ainvoke` produces the
        state final_state that `handle_message_event`'s Phase 2 reads:
        `messages[-1].content` for the reply text and `escalation_reason` for
        the handoff signal. Escalation is inferred from the scripted reply
        starting with 'ESCALATE:' — same contract the real check_escalation
        node enforces.
        """
        recorder = self

        class _FakeGraph:
            async def ainvoke(self, _state):
                recorder.llm_calls += 1
                reply = recorder.reply
                escalation = None
                first_line = reply.strip().splitlines()[0] if reply else ""
                if first_line.upper().startswith("ESCALATE:"):
                    escalation = first_line.split(":", 1)[1].strip() or "(no reason given)"
                return {
                    "messages": [AIMessage(
                        content=reply,
                        usage_metadata={"input_tokens": 5, "output_tokens": 5,
                                        "total_tokens": 10},
                    )],
                    "escalation_reason": escalation,
                }

        return _FakeGraph()

    async def post_message(self, *, account_id, conversation_id, content,
                           private=False):
        (self.private if private else self.public).append(content)
        return {"id": 1}

    async def toggle_status(self, *, account_id, conversation_id, status):
        self.statuses.append(status)
        return {"ok": True}


def run(payload: dict, tenant_id, reply: str = "Sure, happy to help!") -> Recorder:
    """Drive one webhook delivery with the graph/Chatwoot stubbed."""
    rec = Recorder(reply)

    async def post_message(_self, **kw):
        return await rec.post_message(**kw)

    async def toggle_status(_self, **kw):
        return await rec.toggle_status(**kw)

    with patch("app.core.agent.build_graph", rec.build_graph_stub), \
         patch("app.integrations.mori_connect.MoriConnectClient.post_message", post_message), \
         patch("app.integrations.mori_connect.MoriConnectClient.toggle_status", toggle_status):
        from app.core.agent import handle_message_event
        asyncio.run(handle_message_event(payload, tenant_id=tenant_id))
    return rec


def conversation_row() -> Conversation | None:
    with get_session() as db:
        return db.scalar(
            select(Conversation).where(
                Conversation.chatwoot_conversation_id == CONVERSATION_ID
            )
        )


def message_count(role: str) -> int:
    with get_session() as db:
        return db.scalar(
            select(func.count()).select_from(Message).where(Message.role == role)
        )


def main() -> None:
    print("=" * 72)
    print("HANDOFF LIFECYCLE HARNESS")
    print("=" * 72)
    tenant_id = seed_tenant()
    print(f"tenant {tenant_id}\n")
    try:
        _run_cases(tenant_id)
    finally:
        # Always clean up so a passing run doesn't leave the test tenant
        # behind. seed_tenant does the same pre-run cleanup, so this belt-
        # and-braces is just for tidiness on the golden path.
        with get_session() as db:
            cleanup_by_mori_connect_account_id(db, ACCOUNT_ID)


def _run_cases(tenant_id: uuid.UUID) -> None:

    # ─── CASE 1: voluntary escalation ────────────────────────────────────────
    rec = run(incoming("I want to speak to a human"), tenant_id,
              reply="ESCALATE: customer explicitly asked for a person")
    conv = conversation_row()
    leaked = any("ESCALATE:" in c for c in rec.public)
    ok = (
        conv is not None
        and conv.status == "escalated"
        and conv.escalation_reason == "customer explicitly asked for a person"
        and conv.escalated_at is not None
        and len(rec.public) == 1
        and not leaked
        and rec.statuses == ["open"]
    )
    record(
        "CASE 1: ESCALATE marker escalates, hides the marker, opens in Chatwoot",
        ok,
        f"status={conv.status if conv else None!r} "
        f"reason={conv.escalation_reason if conv else None!r} "
        f"customer_saw={rec.public} chatwoot_status={rec.statuses} "
        f"marker_leaked={leaked}",
    )

    # ─── CASE 2: bot stays silent once escalated ─────────────────────────────
    # Chatwoot reports `open` here because CASE 1 just flipped it to `open` on
    # handoff. That is the real post-escalation state: `pending`/`resolved`
    # would mean the admin deliberately handed control back (see CASE 4).
    before = message_count("user")
    rec = run(incoming("are you still there?", chatwoot_status="open"), tenant_id)
    after = message_count("user")
    ok = rec.llm_calls == 0 and rec.public == [] and after == before + 1
    record(
        "CASE 2: escalated conversation — message stored, no LLM, no reply",
        ok,
        f"llm_calls={rec.llm_calls} (want 0), replies={rec.public} (want []), "
        f"user rows {before}->{after} (want +1 for history)",
    )

    # ─── CASE 4 (setup): admin hands control back ────────────────────────────
    # Chatwoot reports `pending`, meaning the admin clicked "Mark as pending".
    rec = run(incoming("hello again?", chatwoot_status="pending"), tenant_id)
    conv = conversation_row()
    ok = (
        conv is not None
        and conv.status != "escalated"
        and conv.escalation_reason is None
        and conv.escalated_at is None
        and rec.llm_calls == 1
        and len(rec.public) == 1
    )
    record(
        "CASE 4: admin hand-back clears escalation and the bot resumes",
        ok,
        f"status={conv.status if conv else None!r} "
        f"reason={conv.escalation_reason if conv else None!r} "
        f"llm_calls={rec.llm_calls} (want 1) replies={len(rec.public)} (want 1)",
    )

    # ─── CASE 3: human agent takeover ────────────────────────────────────────
    rec = run(outgoing("Hi, Sarah here, taking over.", sender_type="user"),
              tenant_id)
    conv = conversation_row()
    ok = (
        conv is not None
        and conv.status == "escalated"
        and conv.escalation_reason == "human agent replied"
        and rec.llm_calls == 0
        and rec.public == []
    )
    record(
        "CASE 3: human agent reply silences the bot",
        ok,
        f"status={conv.status if conv else None!r} "
        f"reason={conv.escalation_reason if conv else None!r} "
        f"llm_calls={rec.llm_calls} (want 0)",
    )

    # ─── CASE 5: the bot's own echo must not count as takeover ───────────────
    # Reset to a non-escalated state first so the assertion is meaningful.
    with get_session() as db:
        c = db.scalar(select(Conversation).where(
            Conversation.chatwoot_conversation_id == CONVERSATION_ID))
        c.status = "pending"
        c.escalated_at = None
        c.escalation_reason = None

    rec = run(outgoing("Sure, happy to help!", sender_type="agent_bot"), tenant_id)
    conv = conversation_row()
    ok = conv is not None and conv.status != "escalated"
    record(
        "CASE 5: bot's own echo is not mistaken for a human takeover",
        ok,
        f"status={conv.status if conv else None!r} (want not 'escalated' — "
        f"'escalated' here is the self-silencing bug)",
    )

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

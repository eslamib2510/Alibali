"""Regression harness for webhook delivery safety.

Run inside the app image with a live Postgres + Redis:

    python -m tests.proof_of_bugs

Covers two failure modes that only appear under real delivery conditions:

  CASE 1: Redis is down when the webhook tries to enqueue. The job never
          reaches ARQ, so ARQ has nothing to retry. The endpoint must answer
          500 so the platform redelivers (it retries agent-bot webhooks on
          429/500); answering 200 drops the message with no safety net.

  CASE 2: the same `message_created` event is delivered twice, which happens
          on ARQ retry and on platform redelivery. The agent must process
          it once: one graph invocation, one reply to the customer.

Gemini and the inbox platform are stubbed. Everything else is real: real
Postgres, real migrations, real HTTP request through the FastAPI app, real
SQLAlchemy sessions.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import patch

from langchain_core.messages import AIMessage
from sqlalchemy import func, select, text

from app.core import crypto
from app.db.models.message import Message
from app.db.models.tenant import Tenant
from app.db.session import get_session
from tests._helpers import cleanup_by_mori_connect_account_id

ACCOUNT_ID = -4242
CONVERSATION_ID = -777
CHATWOOT_MESSAGE_ID = 999001

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {detail}")


def seed_tenant() -> tuple[uuid.UUID, str]:
    """Fresh tenant with an encrypted bot token. Returns (tenant_id, token).

    Scoped to the negative test ACCOUNT_ID so cleanup only touches rows this
    harness created. Real inbox platform account ids are positive.
    """
    token = f"tok_{uuid.uuid4().hex[:16]}"
    with get_session() as db:
        cleanup_by_mori_connect_account_id(db, ACCOUNT_ID)
        tenant = Tenant(
            slug=f"proof-{uuid.uuid4().hex[:8]}",
            name="Proof Tenant",
            mori_connect_account_id=ACCOUNT_ID,
            prompt_template="You are a helpful receptionist.",
            webhook_token=token,
            mori_connect_bot_token_enc=crypto.encrypt("fake-bot-token"),
            notify_admin_on_message=False,
        )
        db.add(tenant)
        db.flush()
        return tenant.id, token


def webhook_payload() -> dict:
    return {
        "event": "message_created",
        "id": CHATWOOT_MESSAGE_ID,
        "content": "Do you have ice baths in stock?",
        "message_type": "incoming",
        "account": {"id": ACCOUNT_ID},
        "conversation": {"id": CONVERSATION_ID, "status": "pending",
                         "custom_attributes": {}},
        "sender": {"id": 55, "name": "Test Customer", "type": "contact"},
        "inbox": {"id": 9, "channel_type": "Channel::Whatsapp"},
    }


# ─── CASE 1: enqueue failure must be reported to Chatwoot ────────────────────


def case_1_enqueue_failure_asks_for_retry(token: str) -> None:
    """Simulate Redis being down at enqueue time and inspect the HTTP reply."""
    from fastapi.testclient import TestClient

    from app.main import app

    class BrokenRedisPool:
        """Stands in for an ARQ pool whose Redis has gone away."""

        def __init__(self) -> None:
            self.calls = 0

        async def enqueue_job(self, *args, **kwargs):
            self.calls += 1
            raise ConnectionError("Redis is down")

        async def close(self):
            return None

    broken = BrokenRedisPool()

    with TestClient(app, raise_server_exceptions=False) as client:
        app.state.redis_pool = broken
        response = client.post(
            f"/api/mori-connect?token={token}", json=webhook_payload()
        )

    status = response.status_code
    ok = status == 500 and broken.calls == 1
    record(
        "CASE 1: enqueue failure returns 500 so the platform redelivers",
        ok,
        f"enqueue raised ConnectionError, endpoint returned HTTP {status} "
        f"(want 500; the platform retries agent-bot webhooks on 429/500).",
    )


# ─── CASE 2: duplicate delivery must be a no-op ──────────────────────────────


def case_2_duplicate_delivery_is_ignored(tenant_id: uuid.UUID) -> None:
    """Deliver the identical event twice, exactly as a retry would."""
    from app.core.agent import handle_message_event

    posted: list[str] = []
    llm_calls: list[str] = []

    def fake_build_graph(*, tenant_id, tenant_prompt):
        class _FakeGraph:
            async def ainvoke(self, _state):
                llm_calls.append(tenant_prompt)
                return {
                    "messages": [AIMessage(
                        content="Yes, we have ice baths in stock!",
                        usage_metadata={"input_tokens": 10, "output_tokens": 8,
                                        "total_tokens": 18},
                    )],
                    "escalation_reason": None,
                }
        return _FakeGraph()

    async def fake_post_message(self, *, account_id, conversation_id, content,
                                private=False):
        if not private:
            posted.append(content)
        return {"id": 1}

    async def fake_toggle_status(self, *, account_id, conversation_id, status):
        return {"ok": True}

    with patch("app.core.agent.build_graph", fake_build_graph), \
         patch("app.integrations.mori_connect.MoriConnectClient.post_message",
               fake_post_message), \
         patch("app.integrations.mori_connect.MoriConnectClient.toggle_status",
               fake_toggle_status):
        payload = webhook_payload()
        # Delivery #1, then the identical redelivery.
        asyncio.run(handle_message_event(payload, tenant_id=tenant_id))
        asyncio.run(handle_message_event(payload, tenant_id=tenant_id))

    with get_session() as db:
        # Scope by tenant_id so residue from prior harness runs (or from
        # other test tenants that happen to share CHATWOOT_MESSAGE_ID)
        # can't skew the count. Dedup itself is per-tenant.
        user_rows = db.scalar(
            select(func.count())
            .select_from(Message)
            .where(Message.tenant_id == tenant_id)
            .where(Message.chatwoot_message_id == CHATWOOT_MESSAGE_ID)
        )
        assistant_rows = db.scalar(
            select(func.count())
            .select_from(Message)
            .where(Message.tenant_id == tenant_id)
            .where(Message.role == "assistant")
        )

    ok = (
        len(llm_calls) == 1
        and len(posted) == 1
        and user_rows == 1
        and assistant_rows == 1
    )
    record(
        "CASE 2: redelivery does not produce a second reply",
        ok,
        f"LLM invoked {len(llm_calls)}x (want 1), replies posted "
        f"{len(posted)}x (want 1), user rows {user_rows} (want 1), "
        f"assistant rows {assistant_rows} (want 1).",
    )

    # The DB-level backstop for the concurrent-delivery race.
    with get_session() as db:
        unique_idx = db.execute(text("""
            SELECT COUNT(*) FROM pg_indexes
            WHERE tablename = 'messages'
              AND indexdef ILIKE '%chatwoot_message_id%'
              AND indexdef ILIKE '%UNIQUE%'
        """)).scalar()
    record(
        "CASE 2b: unique index guards against the concurrent race",
        unique_idx == 1,
        f"unique indexes on messages(tenant_id, chatwoot_message_id): "
        f"{unique_idx} (want 1).",
    )


def main() -> None:
    print("=" * 72)
    print("REGRESSION HARNESS: webhook delivery safety")
    print("=" * 72)

    tenant_id, token = seed_tenant()
    print(f"seeded tenant {tenant_id} token={token}\n")

    case_1_enqueue_failure_asks_for_retry(token)
    print()
    case_2_duplicate_delivery_is_ignored(tenant_id)

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

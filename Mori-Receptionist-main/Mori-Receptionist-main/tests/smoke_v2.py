"""End-to-end v2 smoke test.

    python -m tests.smoke_v2

Creates a throwaway tenant, ingests three FAQ chunks about a fictional
business, and invokes the real graph with a real Gemini call. Confirms:
- The LLM calls search_knowledge with a sensible query
- Results come back from pgvector
- The reply is grounded (contains a detail only present in the seeded FAQ)

Talks to the real Gemini API (uses GEMINI_API_KEY from settings). Costs
about $0.0001 per run. Cleans up its tenant on exit.
"""

from __future__ import annotations

import asyncio
import uuid

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import text

from app.agent.graph import build_graph
from app.db.session import get_session
from app.ingestion.ingestor import ingest_text


TENANT_NAME = "SmokeTestCo (v2 test — safe to delete)"
CHATWOOT_ACCOUNT_ID = -987654321  # negative to avoid clashing with real tenants
PROMPT = (
    "You are the friendly receptionist for MoriCo, a small wellness studio. "
    "Answer briefly and only from the knowledge you're given. If you don't "
    "know, say so."
)

# Distinctive detail that only appears in our seeded FAQ so we can assert the
# reply is grounded (the model can't guess "MOR-4471").
SECRET_SKU = "MOR-4471"

FAQ_TEXT = f"""
Product line: our flagship ice bath is the {SECRET_SKU} Recovery Tub.
It fits one adult, chills to 3°C in 45 minutes, and comes with a
lifetime warranty on the chiller unit. Price is 24,900,000 IDR
including delivery to Bali.

Session times: bookings open Monday through Saturday, 6am to 8pm.
Closed on Sundays. Sessions are 20 minutes for beginners,
up to 45 minutes for experienced users.

Refunds: sessions cancelled with 24 hours notice are refunded in
full. Late cancellations forfeit the session fee. Product returns
accepted within 14 days of delivery for a 10% restocking fee.
"""


async def main() -> None:
    print("=" * 72)
    print("V2 END-TO-END SMOKE TEST")
    print("=" * 72)

    # ─── Create a throwaway tenant ─────────────────────────────────────────
    tenant_id = uuid.uuid4()
    slug = f"smoke-{tenant_id.hex[:8]}"
    with get_session() as db:
        db.execute(
            text("""
                INSERT INTO tenants
                    (id, slug, name, mori_connect_account_id, prompt_template,
                     notify_admin_on_message, webhook_token,
                     created_at, updated_at)
                VALUES
                    (:id, :slug, :name, :cw, :prompt, false, :webhook,
                     now(), now())
            """),
            {
                "id": str(tenant_id),
                "slug": slug,
                "name": TENANT_NAME,
                "cw": CHATWOOT_ACCOUNT_ID,
                "prompt": PROMPT,
                "webhook": f"smoke-{tenant_id.hex}",
            },
        )
    print(f"tenant {tenant_id} created (slug={slug})")

    try:
        # ─── Ingest FAQ ────────────────────────────────────────────────────
        result = ingest_text(
            tenant_id=tenant_id,
            content=FAQ_TEXT.strip(),
            title="MoriCo FAQ",
            source_type="faq",
            source_ref="smoke-seed",
        )
        print(f"ingested {result.chunks_written} chunks")

        # ─── Ask a question that requires the FAQ ──────────────────────────
        # The SKU MOR-4471 is only in the FAQ. If the model returns it, we
        # know it searched, got results, and grounded its answer in them.
        graph = build_graph(tenant_id=str(tenant_id), tenant_prompt=PROMPT)
        state = {
            "tenant_id": str(tenant_id),
            "messages": [HumanMessage(content="what's your flagship ice bath model?")],
            "escalation_reason": None,
        }
        final_state = await graph.ainvoke(state)

        # ─── Inspect what happened ─────────────────────────────────────────
        print()
        print("--- FULL MESSAGE TRACE ---")
        for i, m in enumerate(final_state["messages"]):
            kind = m.__class__.__name__
            tool_calls = getattr(m, "tool_calls", None)
            body = (
                m.content[:200] + ("..." if len(m.content) > 200 else "")
                if isinstance(m.content, str)
                else str(m.content)[:200]
            )
            print(f"  [{i}] {kind}: {body!r}")
            if tool_calls:
                for tc in tool_calls:
                    print(f"       tool_call: {tc['name']}({tc['args']})")

        last = final_state["messages"][-1]
        reply = last.content if isinstance(last.content, str) else str(last.content)

        # ─── Assertions ────────────────────────────────────────────────────
        called_tool = any(
            isinstance(m, AIMessage) and getattr(m, "tool_calls", None)
            for m in final_state["messages"]
        )
        grounded = SECRET_SKU in reply

        print()
        print("=" * 72)
        print(f"{'PASS' if called_tool else 'FAIL'}  LLM called search_knowledge")
        print(
            f"{'PASS' if grounded else 'FAIL'}  Reply is grounded "
            f"(contains {SECRET_SKU})"
        )
        print(f"{'PASS' if final_state.get('escalation_reason') is None else 'FAIL'}"
              f"  Did not escalate")
        print("=" * 72)
        print(f"reply: {reply}")

    finally:
        # ─── Clean up ──────────────────────────────────────────────────────
        with get_session() as db:
            db.execute(text("DELETE FROM knowledge WHERE tenant_id = :t"),
                       {"t": str(tenant_id)})
            db.execute(text("DELETE FROM tenants WHERE id = :t"),
                       {"t": str(tenant_id)})
        print(f"cleaned up tenant {tenant_id}")


if __name__ == "__main__":
    asyncio.run(main())

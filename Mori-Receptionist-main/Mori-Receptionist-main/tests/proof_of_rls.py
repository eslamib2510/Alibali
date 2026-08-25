"""Tenant-isolation harness for the `knowledge` table.

Run inside the app image with a live Postgres:

    python -m tests.proof_of_rls

This is the two-tenant leakage check `docs/rag_plan.md` §2.10 asks for, run
against the real RLS policy rather than trusting the WHERE clauses in
application code.

Checks:
  1. A session scoped to tenant A sees only A's rows — with NO `WHERE
     tenant_id` in the query at all. That is the whole point of RLS: it must
     hold even when application code forgets to filter.
  2. Same for tenant B.
  3. A session that sets no tenant sees nothing (fails closed).
  4. Writing a row for a different tenant than the session is scoped to is
     rejected by Postgres.
"""

from __future__ import annotations

import uuid

from sqlalchemy import text

from app.db.models.tenant import Tenant
from app.db.session import get_session
from tests._helpers import cleanup_by_slug_prefix

TEST_SLUG_PREFIX = "test-rls-"

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {detail}")


def seed() -> tuple[uuid.UUID, uuid.UUID]:
    """Two tenants, one knowledge row each. Returns (tenant_a, tenant_b)."""
    with get_session() as db:
        cleanup_by_slug_prefix(db, TEST_SLUG_PREFIX)
        rows = []
        for label in ("alpha", "beta"):
            t = Tenant(
                slug=f"{TEST_SLUG_PREFIX}{label}-{uuid.uuid4().hex[:8]}",
                name=f"Tenant {label}",
                mori_connect_account_id=abs(hash(label)) % 100000,
                prompt_template="prompt",
                webhook_token=f"tok_{uuid.uuid4().hex[:16]}",
            )
            db.add(t)
            db.flush()
            rows.append(t.id)
        tenant_a, tenant_b = rows

    # Insert one knowledge row per tenant, each inside its own scoped session.
    vector = "[" + ",".join(["0.1"] * 1536) + "]"
    for tid, secret in ((tenant_a, "ALPHA SECRET"), (tenant_b, "BETA SECRET")):
        with get_session(tenant_id=tid) as db:
            db.execute(
                text("""
                    INSERT INTO knowledge
                        (id, tenant_id, source_type, content, embedding)
                    VALUES
                        (gen_random_uuid(), :tid, 'faq', :content,
                         CAST(:emb AS vector))
                """),
                {"tid": str(tid), "content": secret, "emb": vector},
            )
    return tenant_a, tenant_b


def unfiltered_contents(tenant_id) -> list[str]:
    """Deliberately NO `WHERE tenant_id` — RLS alone must scope this."""
    with get_session(tenant_id=tenant_id) as db:
        return [r[0] for r in db.execute(text("SELECT content FROM knowledge"))]


def main() -> None:
    print("=" * 72)
    print("TENANT ISOLATION HARNESS — knowledge table RLS")
    print("=" * 72)

    tenant_a, tenant_b = seed()
    print(f"tenant A = {tenant_a}\ntenant B = {tenant_b}\n")

    a_sees = unfiltered_contents(tenant_a)
    record(
        "Tenant A sees only its own row (no WHERE clause used)",
        a_sees == ["ALPHA SECRET"],
        f"got {a_sees} (want ['ALPHA SECRET'])",
    )

    b_sees = unfiltered_contents(tenant_b)
    record(
        "Tenant B sees only its own row (no WHERE clause used)",
        b_sees == ["BETA SECRET"],
        f"got {b_sees} (want ['BETA SECRET'])",
    )

    with get_session() as db:
        unscoped = [r[0] for r in db.execute(text("SELECT content FROM knowledge"))]
    record(
        "Unscoped session sees nothing (fails closed)",
        unscoped == [],
        f"got {unscoped} (want [])",
    )

    # Writing across tenants must be refused.
    vector = "[" + ",".join(["0.1"] * 1536) + "]"
    try:
        with get_session(tenant_id=tenant_a) as db:
            db.execute(
                text("""
                    INSERT INTO knowledge
                        (id, tenant_id, source_type, content, embedding)
                    VALUES
                        (gen_random_uuid(), :tid, 'faq', 'SMUGGLED',
                         CAST(:emb AS vector))
                """),
                {"tid": str(tenant_b), "emb": vector},
            )
        rejected, why = False, "insert succeeded — cross-tenant write allowed"
    except Exception as e:
        rejected, why = True, f"rejected: {type(e).__name__}"
    record(
        "Session scoped to A cannot write a row owned by B",
        rejected,
        why,
    )

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

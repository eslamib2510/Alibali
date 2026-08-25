"""Ingestion + admin endpoint harness.

    python -m tests.proof_of_ingest

Real Postgres, real RLS, real HTTP through the FastAPI app. Only the Gemini
embedding call is stubbed.

Two things are under test. First the pipeline: does raw text end up as
correctly-scoped, searchable chunks, and does re-ingesting a source replace
its chunks instead of piling up duplicates? Second the gate: this endpoint can
write to ANY tenant's knowledge base, so its auth gets the same scrutiny as
the code behind it.
"""

from __future__ import annotations

import uuid
from unittest.mock import patch

from sqlalchemy import text

from app.db.models.tenant import Tenant
from app.db.session import get_session
from tests._helpers import cleanup_by_slug_prefix

results: list[tuple[str, bool, str]] = []
DIMS = 1536
ADMIN_KEY = "test-admin-key-do-not-use-in-prod"


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {detail}")


def fake_embed_batch(texts, *, task_type=None, dims=DIMS):
    """Deterministic stand-in: one distinct unit vector per input."""
    out = []
    for i, _ in enumerate(texts):
        v = [0.0] * DIMS
        v[i % DIMS] = 1.0
        out.append(v)
    return out


TEST_SLUG_PREFIX = "test-ingest-"


def seed_tenant(label: str) -> uuid.UUID:
    with get_session() as db:
        t = Tenant(
            slug=f"{TEST_SLUG_PREFIX}{label}-{uuid.uuid4().hex[:8]}",
            name=f"Tenant {label}",
            mori_connect_account_id=abs(hash(label + uuid.uuid4().hex)) % 1000000,
            prompt_template="prompt",
            webhook_token=f"tok_{uuid.uuid4().hex[:16]}",
        )
        db.add(t)
        db.flush()
        return t.id


def count_chunks(tenant_id, source_ref: str | None = None) -> int:
    with get_session(tenant_id=tenant_id) as db:
        if source_ref is None:
            return db.execute(
                text("SELECT count(*) FROM knowledge WHERE tenant_id = :t"),
                {"t": str(tenant_id)},
            ).scalar()
        return db.execute(
            text("SELECT count(*) FROM knowledge "
                 "WHERE tenant_id = :t AND source_ref = :r"),
            {"t": str(tenant_id), "r": source_ref},
        ).scalar()


def main() -> None:
    print("=" * 72)
    print("INGESTION + ADMIN ENDPOINT HARNESS")
    print("=" * 72)

    with get_session() as db:
        cleanup_by_slug_prefix(db, TEST_SLUG_PREFIX)

    tenant_a = seed_tenant("alpha")
    tenant_b = seed_tenant("beta")
    print(f"tenant A = {tenant_a}\ntenant B = {tenant_b}\n")

    from app.ingestion import ingestor

    with patch.object(ingestor, "embed_batch", fake_embed_batch):
        # 1. Long text becomes several chunks.
        long_doc = " ".join(
            f"Sentence {i} about our ice bath and sauna recovery services."
            for i in range(1, 200)
        )
        result = ingestor.ingest_text(
            tenant_id=tenant_a, content=long_doc, title="Recovery",
            source_type="faq", source_ref="faq-recovery",
        )
        record(
            "Long document is split into multiple stored chunks",
            result.chunks_written > 1
            and count_chunks(tenant_a, "faq-recovery") == result.chunks_written,
            f"{result.chunks_written} chunks written, "
            f"{count_chunks(tenant_a, 'faq-recovery')} rows in the table",
        )

        # 2. Re-ingesting the same source_ref REPLACES rather than duplicates.
        #    This is what makes a nightly product sync safe to re-run.
        before = count_chunks(tenant_a, "faq-recovery")
        again = ingestor.ingest_text(
            tenant_id=tenant_a, content="A much shorter replacement answer.",
            title="Recovery", source_type="faq", source_ref="faq-recovery",
        )
        after = count_chunks(tenant_a, "faq-recovery")
        record(
            "Re-ingesting a source_ref replaces its chunks, never duplicates",
            after == again.chunks_written and again.chunks_replaced == before
            and after < before,
            f"{before} chunks -> re-ingest replaced {again.chunks_replaced} "
            f"-> {after} now (a shorter source must SHRINK the row count, not "
            f"add to it)",
        )

        # 3. Without source_ref, ingest always adds.
        base = count_chunks(tenant_a)
        ingestor.ingest_text(tenant_id=tenant_a, content="Standalone note one.")
        ingestor.ingest_text(tenant_id=tenant_a, content="Standalone note two.")
        record(
            "Ingest without source_ref accumulates",
            count_chunks(tenant_a) == base + 2,
            f"{base} -> {count_chunks(tenant_a)} after two ref-less ingests",
        )

        # 4. Chunks land under the right tenant only.
        ingestor.ingest_text(tenant_id=tenant_b, content="Beta private note.")
        with get_session(tenant_id=tenant_a) as db:
            a_sees_beta = db.execute(
                text("SELECT count(*) FROM knowledge WHERE content LIKE '%Beta private%'")
            ).scalar()
        record(
            "Ingested chunks are scoped to their tenant",
            a_sees_beta == 0 and count_chunks(tenant_b) == 1,
            f"tenant A sees {a_sees_beta} of B's chunks; B has "
            f"{count_chunks(tenant_b)}",
        )

        # 5. Empty content is a no-op, not a row of nothing.
        empty = ingestor.ingest_text(tenant_id=tenant_a, content="   \n  ")
        record(
            "Empty content writes nothing",
            empty.chunks_written == 0 and empty.chunk_ids == [],
            f"chunks_written={empty.chunks_written}",
        )

        # 6. Invalid source_type is rejected before hitting the CHECK
        #    constraint, so the caller gets a usable message.
        raised = None
        try:
            ingestor.ingest_text(
                tenant_id=tenant_a, content="text", source_type="nonsense"
            )
        except ValueError as e:
            raised = str(e)
        record(
            "Invalid source_type raises a clear error, not an IntegrityError",
            raised is not None and "source_type" in raised,
            f"raised: {raised!r}",
        )

    # ─── HTTP layer ──────────────────────────────────────────────────────────
    from fastapi.testclient import TestClient

    from app.api import deps
    from app.main import app

    auth = {"Authorization": f"Bearer {ADMIN_KEY}"}

    with patch.object(deps.settings, "RECEPTIONIST_ADMIN_API_KEY", ADMIN_KEY), \
         patch.object(ingestor, "embed_batch", fake_embed_batch), \
         TestClient(app, raise_server_exceptions=False) as client:

        # 7. Happy path through HTTP.
        r = client.post("/api/knowledge", headers=auth, json={
            "tenant_id": str(tenant_a),
            "content": "We open at nine in the morning every weekday.",
            "title": "Hours",
            "source_type": "faq",
            "source_ref": "faq-hours",
        })
        body = r.json() if r.status_code < 500 else {}
        record(
            "POST /api/knowledge ingests and reports what it wrote",
            r.status_code == 201 and body.get("chunks_written", 0) >= 1,
            f"HTTP {r.status_code}, body {body}",
        )

        # 8. Unknown tenant is a 404, not a foreign-key explosion.
        r = client.post("/api/knowledge", headers=auth, json={
            "tenant_id": str(uuid.uuid4()),
            "content": "orphan",
        })
        record(
            "Unknown tenant returns 404",
            r.status_code == 404,
            f"HTTP {r.status_code} (want 404)",
        )

        # 9. Listing is scoped to the tenant asked for.
        r = client.get(f"/api/knowledge?tenant_id={tenant_b}", headers=auth)
        items = r.json().get("items", [])
        record(
            "GET /api/knowledge lists only that tenant's chunks",
            r.status_code == 200 and len(items) == 1
            and "Beta private" in items[0]["preview"],
            f"HTTP {r.status_code}, {len(items)} item(s) for tenant B",
        )

        # 10. Delete by source_ref removes the whole source.
        r = client.delete(
            f"/api/knowledge/source/faq-hours?tenant_id={tenant_a}", headers=auth
        )
        record(
            "DELETE by source_ref removes every chunk of that source",
            r.status_code == 200 and r.json()["deleted"] >= 1
            and count_chunks(tenant_a, "faq-hours") == 0,
            f"deleted {r.json().get('deleted')}, remaining "
            f"{count_chunks(tenant_a, 'faq-hours')}",
        )

        # ─── Auth. This key writes to EVERY tenant, so it gets real scrutiny.
        r = client.post("/api/knowledge", json={
            "tenant_id": str(tenant_a), "content": "no auth header",
        })
        record(
            "Missing Authorization header is rejected",
            r.status_code == 401,
            f"HTTP {r.status_code} (want 401)",
        )

        r = client.post("/api/knowledge", headers={"Authorization": "Bearer wrong"},
                        json={"tenant_id": str(tenant_a), "content": "bad key"})
        record(
            "Wrong admin key is rejected",
            r.status_code == 401,
            f"HTTP {r.status_code} (want 401)",
        )

        r = client.get(f"/api/knowledge?tenant_id={tenant_a}")
        record(
            "Read endpoints are gated too, not just writes",
            r.status_code == 401,
            f"HTTP {r.status_code} (want 401)",
        )

    # 14. Unconfigured key must fail CLOSED. A fresh deploy missing the env
    #     var must not expose every tenant's knowledge base.
    with patch.object(deps.settings, "RECEPTIONIST_ADMIN_API_KEY", None), \
         TestClient(app, raise_server_exceptions=False) as client:
        r = client.post("/api/knowledge", headers=auth, json={
            "tenant_id": str(tenant_a), "content": "should not land",
        })
        record(
            "Unset admin key blocks access rather than opening it",
            r.status_code == 503,
            f"HTTP {r.status_code} (want 503 — never 200)",
        )

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

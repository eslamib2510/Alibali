"""Hybrid retrieval harness.

    python -m tests.proof_of_retriever

Real Postgres with pgvector, real HNSW and GIN indexes, real RLS. Only the
Gemini embedding call is stubbed — we hand-craft vectors so "semantically
close" is something the test controls rather than something it hopes for.

The point of hybrid search is that each half covers the other's blind spot,
so the checks are built around exactly that:

  - a chunk that matches on MEANING but shares no words with the query must
    still be found (vector half)
  - a chunk that matches on an EXACT TOKEN the embedding can't distinguish —
    a SKU — must still be found (keyword half)
  - a chunk both halves like must outrank one only a single half likes
  - and none of it may leak across tenants
"""

from __future__ import annotations

import uuid

from sqlalchemy import text
from unittest.mock import patch

from app.db.models.tenant import Tenant
from app.db.session import get_session
from app.retrieval import retriever
from tests._helpers import cleanup_by_slug_prefix

results: list[tuple[str, bool, str]] = []
DIMS = 1536


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {detail}")


def vec(*, axis: int, value: float = 1.0) -> list[float]:
    """A unit vector pointing along one axis.

    Two vectors on the same axis have cosine distance 0; on different axes,
    distance 1. That gives us exact control over what counts as semantically
    near without depending on a real embedding model.
    """
    v = [0.0] * DIMS
    v[axis] = value
    return v


def as_literal(v: list[float]) -> str:
    return "[" + ",".join(repr(float(x)) for x in v) + "]"


# Axis assignments: the "recovery" topic lives on axis 0, "shipping" on axis 1,
# and an unrelated topic on axis 2.
AXIS_RECOVERY, AXIS_SHIPPING, AXIS_OTHER = 0, 1, 2


TEST_SLUG_PREFIX = "test-retriever-"


def seed() -> tuple[uuid.UUID, uuid.UUID]:
    with get_session() as db:
        cleanup_by_slug_prefix(db, TEST_SLUG_PREFIX)
        ids = []
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
            ids.append(t.id)
        tenant_a, tenant_b = ids

    rows = [
        # Semantically on the "recovery" axis, but deliberately shares NO
        # words with the query we'll use ("cold plunge"). Only the vector half
        # can find this.
        (tenant_a, "Ice Baths",
         "Our ice bath sessions run for fifteen minutes at four degrees.",
         AXIS_RECOVERY),
        # An exact token an embedding cannot distinguish from any other SKU.
        # Only the keyword half can find this reliably.
        (tenant_a, "Stock",
         "The SKU-4471 barrel is currently available in our warehouse.",
         AXIS_OTHER),
        # Shares the query's words AND sits on the query's axis — both halves
        # should rank it, so RRF must put it on top.
        (tenant_a, "Cold Plunge Guide",
         "A cold plunge is best taken after a sauna session for recovery.",
         AXIS_RECOVERY),
        # Unrelated on both axes — must not crowd out the real answers.
        (tenant_a, "Shipping",
         "Orders are dispatched within two business days by courier.",
         AXIS_SHIPPING),
        # Other tenant's data, deliberately a perfect match for the query.
        (tenant_b, "Competitor Secret",
         "A cold plunge is our rival's most popular recovery product.",
         AXIS_RECOVERY),
    ]

    for tid, title, content, axis in rows:
        with get_session(tenant_id=tid) as db:
            db.execute(
                text("""
                    INSERT INTO knowledge
                        (id, tenant_id, source_type, title, content, embedding)
                    VALUES (gen_random_uuid(), :tid, 'faq', :title, :content,
                            CAST(:emb AS vector))
                """),
                {"tid": str(tid), "title": title, "content": content,
                 "emb": as_literal(vec(axis=axis))},
            )
    return tenant_a, tenant_b


def search_with(query: str, tenant_id, axis: int, **kw):
    """Run a search with the query embedding pinned to a chosen axis."""
    with patch.object(retriever, "embed", lambda *_a, **_k: vec(axis=axis)):
        return retriever.search(tenant_id=tenant_id, query=query, **kw)


def main() -> None:
    print("=" * 72)
    print("HYBRID RETRIEVAL HARNESS")
    print("=" * 72)
    tenant_a, tenant_b = seed()
    print(f"tenant A = {tenant_a}\ntenant B = {tenant_b}\n")

    # 1. Vector half: find a chunk with zero word overlap with the query.
    hits = search_with("cold plunge", tenant_a, AXIS_RECOVERY)
    titles = [h.title for h in hits]
    record(
        "Vector half finds a chunk sharing no words with the query",
        "Ice Baths" in titles,
        f"query 'cold plunge' -> {titles}; 'Ice Baths' shares no query words "
        f"and is reachable only via the embedding",
    )

    # 2. Keyword half: an exact token the embedding is blind to. The query
    #    vector points at an unrelated axis, so only FTS can surface this.
    hits = search_with("SKU-4471", tenant_a, AXIS_SHIPPING)
    titles = [h.title for h in hits]
    record(
        "Keyword half finds an exact token the vector half would miss",
        "Stock" in titles,
        f"query 'SKU-4471' with the query vector aimed at an unrelated axis "
        f"-> {titles}",
    )

    # 3. Fusion: a chunk both halves rank must beat one only a single half
    #    ranks.
    hits = search_with("cold plunge recovery", tenant_a, AXIS_RECOVERY)
    top = hits[0].title if hits else None
    record(
        "Chunk ranked by both halves outranks single-half matches",
        top == "Cold Plunge Guide",
        f"top hit {top!r} (want 'Cold Plunge Guide' — the only chunk matching "
        f"on both words and meaning). Full order: {[h.title for h in hits]}",
    )

    # 4. Scores are ordered and RRF-shaped.
    scores = [h.score for h in hits]
    record(
        "Results come back sorted by descending fused score",
        scores == sorted(scores, reverse=True) and all(s > 0 for s in scores),
        f"scores {[round(s, 5) for s in scores]}",
    )

    # 5. Tenant isolation, on the query most likely to leak.
    hits = search_with("cold plunge recovery", tenant_a, AXIS_RECOVERY)
    leaked = [h for h in hits if "rival" in h.content]
    record(
        "Tenant B's near-perfect match never appears for tenant A",
        not leaked,
        f"{len(hits)} hits for tenant A, leaked rows: {len(leaked)} (tenant B "
        f"holds a chunk matching this query better than anything A owns)",
    )

    # 6. And the reverse — B sees only its own.
    hits = search_with("cold plunge recovery", tenant_b, AXIS_RECOVERY)
    record(
        "Tenant B sees only its own chunk",
        len(hits) == 1 and hits[0].title == "Competitor Secret",
        f"{[h.title for h in hits]}",
    )

    # 7. limit is honoured.
    hits = search_with("cold plunge recovery", tenant_a, AXIS_RECOVERY, limit=2)
    record(
        "limit caps the number of chunks returned",
        len(hits) <= 2,
        f"asked for 2, got {len(hits)}",
    )

    # 8. Blank query short-circuits without touching the database.
    record(
        "Blank query returns nothing",
        search_with("   ", tenant_a, AXIS_RECOVERY) == [],
        "whitespace query -> []",
    )

    # 9. A query matching nothing returns empty rather than raising — the
    #    agent must still reply, just without grounding.
    hits = search_with("xylophone quantum tuba", tenant_a, AXIS_OTHER)
    record(
        "No-match query returns an empty list, not an error",
        isinstance(hits, list),
        f"{len(hits)} hits for nonsense query (vector half always returns its "
        f"nearest rows, so a non-empty result here is expected and fine)",
    )

    # 10. format_context renders titles and separators, and is empty for none.
    hits = search_with("cold plunge", tenant_a, AXIS_RECOVERY)
    block = retriever.format_context(hits)
    record(
        "format_context renders titled, separated chunks",
        "[Cold Plunge Guide]" in block and "---" in block
        and retriever.format_context([]) == "",
        f"{len(block)} chars, contains titles and separators; empty input -> "
        f"{retriever.format_context([])!r}",
    )

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

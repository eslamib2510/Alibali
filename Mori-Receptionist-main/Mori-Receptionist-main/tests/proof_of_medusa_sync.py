"""Product sync harness.

    python -m tests.proof_of_medusa_sync

Real Postgres, real ingestion pipeline, real chunker. Only the Medusa HTTP
client and the Gemini embedding call are stubbed — we hand-craft product
dicts so shape assumptions are things this test controls.

What's under test:

  - A first-time sync writes one `knowledge` row per product with
    source_type='product' and source_ref = the Medusa id.
  - Re-syncing the same id with a changed description REPLACES the old chunk
    (delete-then-insert on source_ref) — new text present, old text absent,
    per-product row count unchanged.
  - A product that was present last run but not this one gets purged. Without
    that step the RAG store drifts and starts citing products the storefront
    no longer sells.
  - Pagination pulls page 2 when page 1 fills the limit — a real catalogue
    won't fit in one page.
  - format_product bakes the SKU and handle into the embedded text so the
    FTS half of hybrid retrieval can pull the exact-token queries embeddings
    can't distinguish.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import patch

from sqlalchemy import text

from app.db.models.tenant import Tenant
from app.db.session import get_session
from app.services import medusa_sync
from tests._helpers import cleanup_by_slug_prefix

results: list[tuple[str, bool, str]] = []
DIMS = 1536


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {detail}")


def fake_embed_batch(texts, *, task_type=None, dims=DIMS):
    """One distinct unit vector per input — matches proof_of_ingest.py."""
    out = []
    for i, _ in enumerate(texts):
        v = [0.0] * DIMS
        v[i % DIMS] = 1.0
        out.append(v)
    return out


TEST_SLUG_PREFIX = "test-medsync-"


def seed_tenant() -> uuid.UUID:
    with get_session() as db:
        t = Tenant(
            slug=f"{TEST_SLUG_PREFIX}{uuid.uuid4().hex[:8]}",
            name="Product Sync Test",
            mori_connect_account_id=abs(hash("psync" + uuid.uuid4().hex)) % 1000000,
            prompt_template="prompt",
            webhook_token=f"tok_{uuid.uuid4().hex[:16]}",
            medusa_api_url="http://fake-medusa.test",
            # crypto.get_medusa_key returns None if this is empty; a real
            # ciphertext is unnecessary because we patch out the client
            # constructor. Sentinel value keeps the null check happy.
            medusa_api_key_enc=None,
        )
        db.add(t)
        db.flush()
        return t.id


def make_product(pid: str, title: str, sku: str, description: str, *, price: int = 100000,
                 handle: str | None = None) -> dict:
    """Product dict shaped like Medusa v2's /store/products response."""
    return {
        "id": pid,
        "title": title,
        "handle": handle or pid.replace("prod_", ""),
        "subtitle": None,
        "description": description,
        "status": "published",
        "thumbnail": f"https://cdn.test/{pid}.jpg",
        "tags": [{"id": "tag_1", "value": "recovery"}],
        "categories": [{"id": "cat_1", "name": "Wellness"}],
        "collection": None,
        "images": [{"id": f"img_{pid}", "url": f"https://cdn.test/{pid}.jpg", "rank": 0}],
        "variants": [
            {
                "id": f"var_{pid}",
                "title": "Default",
                "sku": sku,
                "options": [],
                "calculated_price": {
                    "id": "price_1",
                    "calculated_amount": price,
                    "currency_code": "idr",
                    "original_amount": price,
                },
            }
        ],
    }


class FakeMedusa:
    """Stands in for MedusaClient inside the sync run.

    `pages` is a list of `(products, total_count)` tuples returned in order
    by successive list_products calls. list_regions returns a single fixed
    region so the sync picks a region_id and threads it through.
    """

    def __init__(self, pages: list[tuple[list[dict], int]]):
        self._pages = list(pages)
        self.list_products_calls: list[dict] = []

    async def list_regions(self):
        return [{"id": "reg_test", "name": "Test", "currency_code": "idr"}]

    async def list_products(self, *, limit, offset, region_id=None, **kw):
        self.list_products_calls.append(
            {"limit": limit, "offset": offset, "region_id": region_id}
        )
        if not self._pages:
            return [], 0
        return self._pages.pop(0)


def count_product_rows(tenant_id, source_ref: str | None = None) -> int:
    with get_session(tenant_id=tenant_id) as db:
        if source_ref is None:
            return db.execute(
                text("SELECT count(*) FROM knowledge WHERE tenant_id = :t "
                     "AND source_type = 'product'"),
                {"t": str(tenant_id)},
            ).scalar()
        return db.execute(
            text("SELECT count(*) FROM knowledge WHERE tenant_id = :t "
                 "AND source_type = 'product' AND source_ref = :r"),
            {"t": str(tenant_id), "r": source_ref},
        ).scalar()


def fetch_content(tenant_id, source_ref: str) -> str:
    with get_session(tenant_id=tenant_id) as db:
        rows = db.execute(
            text("SELECT content FROM knowledge WHERE tenant_id = :t "
                 "AND source_ref = :r ORDER BY created_at"),
            {"t": str(tenant_id), "r": source_ref},
        ).all()
    return "\n---\n".join(r[0] for r in rows)


def load_tenant(tenant_id):
    from sqlalchemy import select
    with get_session() as db:
        t = db.scalar(select(Tenant).where(Tenant.id == tenant_id))
        db.expunge(t)
        return t


def run_sync(tenant, fake: FakeMedusa):
    """Invoke sync_tenant_products with the fake client patched in.

    Also patches out crypto so we don't have to set up a real Fernet key,
    and swaps embed_batch on the ingestor so no HTTP goes out.
    """
    from app.ingestion import ingestor as ingestor_mod
    with patch.object(medusa_sync, "MedusaClient", lambda **_kw: fake), \
         patch.object(medusa_sync.crypto, "get_medusa_key", lambda t: "pk_test"), \
         patch.object(ingestor_mod, "embed_batch", fake_embed_batch):
        return asyncio.run(medusa_sync.sync_tenant_products(tenant))


def main() -> None:
    print("=" * 72)
    print("PRODUCT SYNC HARNESS")
    print("=" * 72)

    with get_session() as db:
        cleanup_by_slug_prefix(db, TEST_SLUG_PREFIX)

    tenant_id = seed_tenant()
    tenant = load_tenant(tenant_id)
    print(f"tenant = {tenant_id}\n")

    # ─── 1. format_product bakes distinctive tokens into the content ────────
    p_shape = make_product(
        "prod_shape", "Shape Test Ice Bath", "MOR-SHAPE-01",
        "A distinctive description string with unique words.",
        handle="shape-test-ice-bath",
    )
    title, content = medusa_sync.format_product(p_shape)
    record(
        "format_product includes SKU and handle in embedded content",
        "MOR-SHAPE-01" in content and "shape-test-ice-bath" in content
        and "Shape Test Ice Bath" == title,
        f"title={title!r}, content contains SKU={('MOR-SHAPE-01' in content)}, "
        f"handle={('shape-test-ice-bath' in content)}, "
        f"price={('100,000' in content or '100000' in content)}",
    )

    # ─── 2. Fresh sync inserts one row per product ──────────────────────────
    p1 = make_product("prod_1", "Ice Bath A", "MOR-A", "Original description one.")
    p2 = make_product("prod_2", "Sauna B", "MOR-B", "Original description two.")
    fake = FakeMedusa(pages=[([p1, p2], 2)])
    result = run_sync(tenant, fake)
    record(
        "Fresh sync writes one knowledge row per product with source_type=product",
        result.products_ingested == 2
        and count_product_rows(tenant_id) == 2
        and count_product_rows(tenant_id, "prod_1") == 1
        and count_product_rows(tenant_id, "prod_2") == 1
        and result.products_deleted == 0,
        f"ingested={result.products_ingested}, chunks={result.chunks_written}, "
        f"rows_total={count_product_rows(tenant_id)}, "
        f"prod_1_rows={count_product_rows(tenant_id, 'prod_1')}",
    )

    # ─── 3. Re-sync with a changed description replaces the chunk ───────────
    before_total = count_product_rows(tenant_id)
    before_p1 = count_product_rows(tenant_id, "prod_1")
    p1_new = make_product(
        "prod_1", "Ice Bath A", "MOR-A",
        "Rewritten body text featuring the unmistakable phrase FLUXCAPACITOR.",
    )
    fake = FakeMedusa(pages=[([p1_new, p2], 2)])
    result = run_sync(tenant, fake)
    after_total = count_product_rows(tenant_id)
    after_p1 = count_product_rows(tenant_id, "prod_1")
    p1_content = fetch_content(tenant_id, "prod_1")
    record(
        "Re-syncing an existing product replaces its chunks in place",
        after_total == before_total
        and after_p1 == before_p1
        and "FLUXCAPACITOR" in p1_content
        and "Original description one" not in p1_content,
        f"total {before_total}->{after_total} (should stay), "
        f"prod_1 rows {before_p1}->{after_p1} (should stay), "
        f"new phrase present={('FLUXCAPACITOR' in p1_content)}, "
        f"old phrase absent={('Original description one' not in p1_content)}",
    )

    # ─── 4. Product missing from this run is deleted (reconciliation) ───────
    fake = FakeMedusa(pages=[([p1_new], 1)])  # p2 has vanished upstream
    result = run_sync(tenant, fake)
    record(
        "Product removed upstream is deleted from knowledge on next sync",
        count_product_rows(tenant_id, "prod_2") == 0
        and count_product_rows(tenant_id, "prod_1") >= 1
        and result.products_deleted == 1,
        f"prod_2 rows now {count_product_rows(tenant_id, 'prod_2')} (want 0), "
        f"prod_1 rows {count_product_rows(tenant_id, 'prod_1')} (want >=1), "
        f"products_deleted reported={result.products_deleted}",
    )

    # ─── 5. Pagination: full page triggers a second fetch ───────────────────
    # Rebuild state cleanly for the pagination check. The DELETE needs the
    # RLS session variable set or the policy blocks it (mori_app has no
    # BYPASSRLS, so a bare DELETE would fail on the USING check silently).
    with get_session(tenant_id=tenant_id) as db:
        db.execute(text("DELETE FROM knowledge WHERE tenant_id = :t"),
                   {"t": str(tenant_id)})
    page_size = medusa_sync.PAGE_SIZE
    page1 = [make_product(f"pg_{i}", f"Prod {i}", f"SKU-{i}", f"body {i}")
             for i in range(page_size)]
    page2 = [make_product(f"pg_x", "Prod X", "SKU-X", "last one")]
    fake = FakeMedusa(pages=[(page1, page_size + 1), (page2, page_size + 1)])
    result = run_sync(tenant, fake)
    record(
        "Pagination pulls page 2 when page 1 fills the limit",
        len(fake.list_products_calls) >= 2
        and fake.list_products_calls[0]["offset"] == 0
        and fake.list_products_calls[1]["offset"] == page_size
        and result.products_ingested == page_size + 1
        and count_product_rows(tenant_id) == page_size + 1,
        f"list_products called {len(fake.list_products_calls)} times, "
        f"offsets={[c['offset'] for c in fake.list_products_calls]}, "
        f"ingested={result.products_ingested}",
    )

    # ─── 6. Region is threaded through: prices land in embedded text ────────
    # (Sanity check that the region_id gets forwarded to list_products.)
    with get_session(tenant_id=tenant_id) as db:
        db.execute(text("DELETE FROM knowledge WHERE tenant_id = :t"),
                   {"t": str(tenant_id)})
    priced = make_product("prod_priced", "Priced Item", "SKU-PRICED",
                          "Costs money.", price=1234567)
    fake = FakeMedusa(pages=[([priced], 1)])
    result = run_sync(tenant, fake)
    body = fetch_content(tenant_id, "prod_priced")
    record(
        "Sync forwards a region_id and price lands in the embedded text",
        fake.list_products_calls[0]["region_id"] == "reg_test"
        and ("1,234,567" in body or "1234567" in body),
        f"region_id passed={fake.list_products_calls[0]['region_id']}, "
        f"price token present={('1,234,567' in body)}",
    )

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

"""Sync a tenant's Medusa product catalogue into the `knowledge` table.

The receptionist doesn't call Medusa at reply time. Instead we periodically
mirror published products into RAG so a customer asking "how much is the ice
bath" gets an answer from the same retrieval path that answers "do you refund
sessions." One query surface, one grounding story.

Design notes
------------
- Ingestion goes through `ingest_text` so chunk sizing, embeddings and
  delete-then-insert on `source_ref` behave the same as an admin-uploaded FAQ.
  A product's `source_ref` is its Medusa id, which is stable across syncs and
  gives us idempotency for free.
- Prices ARE embedded. A customer asking "what does it cost" won't trigger a
  live-price tool call — they'll trigger a knowledge search. If the current
  price isn't in the text, the model has nothing to ground on and either
  guesses or refuses. Price staleness (a day or two, between re-syncs) is
  acceptable; total absence is not.
- Stock quantities are DELIBERATELY NOT embedded. They change every hour and
  a stale "in stock" statement in the RAG store would mislead a customer into
  buying something we can't ship. Live inventory is a future v4 tool call
  (`get_product_stock`), not a RAG concern.
- Reconciliation: a nightly sync must delete rows for products that were
  removed upstream, or the knowledge base drifts and starts referencing
  products the storefront no longer sells. We track ids seen this run and
  purge everything else under `source_type='product'` for this tenant.
- Pagination is one page at a time. A large catalogue could easily exceed
  10k products; buffering the whole list means the embed API's rate limits
  and Postgres' round trips both interleave with fetches naturally.
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any

from sqlalchemy import text

from app.core import crypto
from app.db.session import get_session
from app.integrations.medusa import MedusaClient
from app.ingestion import ingestor

logger = logging.getLogger(__name__)

# Page size for the /store/products call. 100 is Medusa's typical soft cap
# for a query with heavy field expansion — much bigger and the request
# starts timing out on cold caches.
PAGE_SIZE = 100


@dataclass
class SyncResult:
    """Summary of one sync run. Returned to the CLI for logging."""

    products_seen: int = 0
    products_ingested: int = 0
    chunks_written: int = 0
    products_deleted: int = 0
    # Non-fatal per-product failures — the sync keeps going and reports them
    # at the end rather than aborting the whole run on a single bad row.
    errors: list[str] = field(default_factory=list)


def _price_line(variant: dict[str, Any]) -> str | None:
    """One variant's price as human-readable text, or None if no price.

    Medusa's `calculated_price` is only populated when a region_id was on the
    products query; a variant with no price still gets embedded (title + SKU)
    but the price line is skipped rather than filled with "None".
    """
    cp = variant.get("calculated_price") or {}
    amount = cp.get("calculated_amount")
    if amount is None:
        return None
    currency = (cp.get("currency_code") or "").upper()
    # Medusa stores amounts as decimal-scaled numbers, not minor units, so
    # 24900000 is 24,900,000 in the currency — no /100 conversion.
    return f"{amount:,.0f} {currency}".strip()


def _tags(product: dict[str, Any]) -> list[str]:
    return [t.get("value") for t in (product.get("tags") or []) if t.get("value")]


def _categories(product: dict[str, Any]) -> list[str]:
    return [c.get("name") for c in (product.get("categories") or []) if c.get("name")]


def _images(product: dict[str, Any]) -> list[str]:
    urls = [i.get("url") for i in (product.get("images") or []) if i.get("url")]
    if not urls and product.get("thumbnail"):
        urls = [product["thumbnail"]]
    return urls


def format_product(product: dict[str, Any]) -> tuple[str, str]:
    """Render one Medusa product as (title, content) for `ingest_text`.

    The blob is deliberately dense with distinctive tokens — SKUs, handle,
    tag values — because those are exactly what the FTS half of hybrid
    retrieval matches on. An embedding of "SKU-4471" looks like every other
    SKU; only the tsvector index can pull the right row for a query naming
    that code.
    """
    title = (product.get("title") or "").strip() or "(untitled product)"
    handle = (product.get("handle") or "").strip()
    subtitle = (product.get("subtitle") or "").strip()
    description = (product.get("description") or "").strip()

    lines: list[str] = [title]
    if subtitle:
        lines.append(subtitle)
    if handle:
        # Handle doubles as a URL slug and a stable customer-facing id — worth
        # having in the embedded text so a query citing the storefront URL
        # can find the product.
        lines.append(f"Handle: {handle}")

    tags = _tags(product)
    if tags:
        lines.append("Tags: " + ", ".join(tags))

    categories = _categories(product)
    if categories:
        lines.append("Category: " + ", ".join(categories))

    collection = (product.get("collection") or {}).get("title")
    if collection:
        lines.append(f"Collection: {collection}")

    if description:
        lines.append("")
        lines.append(description)

    # Variants and prices. One line per variant so the FTS index can hit
    # individual SKUs directly. Skipping stock quantities is a policy call
    # documented at module top.
    variants = product.get("variants") or []
    if variants:
        lines.append("")
        lines.append("Variants:")
        for v in variants:
            v_title = (v.get("title") or "").strip()
            sku = (v.get("sku") or "").strip()
            price = _price_line(v)
            parts: list[str] = []
            if v_title:
                parts.append(v_title)
            if sku:
                parts.append(f"SKU {sku}")
            if price:
                parts.append(f"price {price}")
            if parts:
                lines.append("- " + " — ".join(parts))

    content = "\n".join(lines).strip()
    return title, content


async def _pick_region_id(client: MedusaClient) -> str | None:
    """Fetch the store's regions and return the first one's id.

    Prices don't populate on the products endpoint without a region_id. We
    pick one region and stick with it for the whole sync — mixing regions
    across products would put IDR and MYR amounts side by side in retrieval
    with no way to tell which was which. If the store has no regions at all
    (fresh Medusa install), we sync without prices rather than crashing.
    """
    try:
        regions = await client.list_regions()
    except Exception as e:
        logger.warning("Could not fetch Medusa regions (%s) — syncing without prices", e)
        return None
    if not regions:
        logger.warning("Medusa store has no regions — syncing without prices")
        return None
    return regions[0].get("id")


def _existing_product_refs(tenant_id) -> set[str]:
    """Every source_ref currently in `knowledge` for source_type='product'.

    Used for reconciliation: anything here that this sync run didn't touch
    is a delisted / deleted product and gets purged at the end.
    """
    with get_session(tenant_id=tenant_id) as db:
        rows = db.execute(
            text("""
                SELECT DISTINCT source_ref
                FROM knowledge
                WHERE tenant_id = :t AND source_type = 'product'
                  AND source_ref IS NOT NULL
            """),
            {"t": str(tenant_id)},
        ).all()
    return {r[0] for r in rows}


async def sync_tenant_products(tenant) -> SyncResult:
    """Pull every published product from Medusa and mirror it into `knowledge`.

    Args:
        tenant: a `Tenant` row with `medusa_api_url` and `medusa_api_key_enc`
            populated. Raises `ValueError` if either is missing — a chat-only
            tenant should never be passed here in the first place.

    Returns:
        SyncResult with counts for logging / CI.
    """
    api_key = crypto.get_medusa_key(tenant)
    if not tenant.medusa_api_url or not api_key:
        raise ValueError(
            f"Tenant {tenant.slug!r} has no Medusa configuration "
            "(medusa_api_url or medusa_api_key_enc is missing)."
        )

    client = MedusaClient(base_url=tenant.medusa_api_url, api_key=api_key)
    result = SyncResult()

    # Snapshot what's already ingested BEFORE we start writing, so
    # reconciliation compares the pre-sync world against the just-seen set
    # rather than deleting rows we just re-inserted.
    previous_refs = _existing_product_refs(tenant.id)
    seen_refs: set[str] = set()

    region_id = await _pick_region_id(client)
    if region_id:
        logger.info("Sync for tenant=%s using region_id=%s", tenant.slug, region_id)

    offset = 0
    page = 0
    while True:
        page += 1
        products, total = await client.list_products(
            limit=PAGE_SIZE, offset=offset, region_id=region_id
        )
        if not products:
            break

        for product in products:
            pid = product.get("id")
            if not pid:
                # No id means we can't compute an idempotency key. Skip loud
                # so a malformed payload is visible rather than silently
                # multiplying chunks on every re-sync.
                result.errors.append("product with no id — skipped")
                continue

            title, content = format_product(product)
            if not content.strip():
                result.errors.append(f"{pid}: empty content — skipped")
                continue

            try:
                # ingest_text is blocking (embed call + inserts); dispatch off
                # the loop so we don't stall other async work in the same
                # process. Idempotent on (tenant_id, source_ref): re-syncing
                # the same product deletes its old chunks and writes new ones.
                ingest = await asyncio.to_thread(
                    ingestor.ingest_text,
                    tenant_id=tenant.id,
                    content=content,
                    title=title,
                    source_type="product",
                    source_ref=pid,
                    metadata={
                        "handle": product.get("handle"),
                        "images": _images(product),
                        "status": product.get("status"),
                    },
                )
            except Exception as e:  # pragma: no cover - logged, not swallowed
                logger.exception("Failed to ingest product %s", pid)
                result.errors.append(f"{pid}: {e}")
                continue

            seen_refs.add(pid)
            result.products_seen += 1
            result.products_ingested += 1
            result.chunks_written += ingest.chunks_written

        logger.info(
            "synced %d products (page %d, offset %d, total ~%d)",
            len(products), page, offset, total,
        )

        offset += len(products)
        # Stop when we've fetched everything Medusa says exists. Fallback:
        # a short page also means we're done, in case `count` is unreliable.
        if offset >= total or len(products) < PAGE_SIZE:
            break

    # Reconciliation: purge products that disappeared upstream. This is the
    # step that keeps the RAG store from accumulating dead products across
    # nightly syncs.
    stale = previous_refs - seen_refs
    for ref in stale:
        try:
            deleted = ingestor.delete_source(tenant_id=tenant.id, source_ref=ref)
            if deleted:
                result.products_deleted += 1
        except Exception as e:  # pragma: no cover - logged
            logger.exception("Failed to delete stale product %s", ref)
            result.errors.append(f"delete {ref}: {e}")

    logger.info(
        "sync complete: tenant=%s seen=%d ingested=%d chunks=%d deleted=%d errors=%d",
        tenant.slug,
        result.products_seen,
        result.products_ingested,
        result.chunks_written,
        result.products_deleted,
        len(result.errors),
    )
    return result

"""Sync one tenant's Medusa product catalogue into `knowledge`.

    python -m scripts.sync_medusa --tenant alabali

Looks up the tenant by slug, pulls every published product from Medusa,
embeds them into the RAG store, and reconciles by deleting products that
disappeared upstream since the last run.

Idempotent — safe to run on a cron. See `app/services/medusa_sync.py` for
the delete-then-insert semantics that make repeated runs behave.
"""

from __future__ import annotations

import argparse
import asyncio

from sqlalchemy import select

from app.db.models.tenant import Tenant
from app.db.session import get_session
from app.services.medusa_sync import sync_tenant_products


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--tenant", required=True, help="Tenant slug (e.g. 'alabali')")
    return p.parse_args()


async def main() -> None:
    args = parse_args()

    # Read the tenant in one short-lived session, then hand it to the sync
    # worker. The sync itself opens its own sessions per write, so we don't
    # hold this one across the (potentially many) embed+insert round trips.
    with get_session() as db:
        tenant = db.scalar(select(Tenant).where(Tenant.slug == args.tenant))
        if tenant is None:
            raise SystemExit(f"Tenant not found: {args.tenant!r}")
        # Force-load the encrypted fields while the session is live, then
        # detach so the object is safe to use after the session closes.
        _ = tenant.medusa_api_url, tenant.medusa_api_key_enc, tenant.slug, tenant.id
        db.expunge(tenant)

    # Fail loud when the tenant isn't configured for Medusa. A chat-only
    # tenant shouldn't be silently skipped — that hides a real config bug
    # (either the tenant is misconfigured or the wrong slug was passed).
    if not tenant.medusa_api_url or not tenant.medusa_api_key_enc:
        raise SystemExit(
            f"Tenant {args.tenant!r} has no Medusa config. Set "
            "medusa_api_url and medusa_api_key_enc on the tenants row "
            "(publishable key, pk_...) and try again."
        )

    result = await sync_tenant_products(tenant)

    print("=" * 60)
    print(f"Sync complete for tenant={args.tenant}")
    print(f"  products seen:      {result.products_seen}")
    print(f"  products ingested:  {result.products_ingested}")
    print(f"  chunks written:     {result.chunks_written}")
    print(f"  products deleted:   {result.products_deleted}")
    if result.errors:
        print(f"  errors ({len(result.errors)}):")
        for e in result.errors:
            print(f"    - {e}")
    print("=" * 60)
    raise SystemExit(0 if not result.errors else 1)


if __name__ == "__main__":
    asyncio.run(main())

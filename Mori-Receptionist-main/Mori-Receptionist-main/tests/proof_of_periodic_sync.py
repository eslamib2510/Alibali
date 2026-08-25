"""Cron wrapper harness for the periodic Medusa sync.

Run inside the app image with a live Postgres:

    python -m tests.proof_of_periodic_sync

`proof_of_medusa_sync` already covers `sync_tenant_products` itself.
This harness covers the ARQ cron wrapper `sync_all_medusa_tenants`
in `app/workers/receptionist.py`, which adds two things on top:

  CASE 1  Filter. Only tenants where BOTH medusa_api_url and
          medusa_api_key_enc are set should be synced. Tenants with
          url-only or key-only are still in the middle of onboarding
          and would otherwise produce a noisy "misconfigured" error
          every tick.

  CASE 2  Error isolation. One tenant whose sync raises must not
          prevent later tenants from being synced. The exception has
          to be caught, logged, and the loop continued.

  CASE 3  Empty-catalogue safety. A tick with zero matching tenants
          must be a fast no-op, not an exception.

The Medusa HTTP client is stubbed at `sync_tenant_products` level so
this harness never touches a live store. Cleanup is scoped by slug
prefix so real tenants are never affected.
"""

from __future__ import annotations

import asyncio
import uuid
from unittest.mock import patch

from app.core import crypto
from app.db.models.tenant import Tenant
from app.db.session import get_session
from app.services.medusa_sync import SyncResult
from app.workers.receptionist import sync_all_medusa_tenants
from tests._helpers import cleanup_by_slug_prefix

TEST_SLUG_PREFIX = "test-periodic-"

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {detail}")


def _make_tenant(*, slug: str, url: str | None, key: str | None) -> uuid.UUID:
    with get_session() as db:
        t = Tenant(
            slug=slug,
            name=slug,
            mori_connect_account_id=abs(hash(slug)) % 1000000,
            prompt_template="prompt",
            webhook_token=f"tok_{uuid.uuid4().hex[:16]}",
            medusa_api_url=url,
            medusa_api_key_enc=crypto.encrypt_optional(key) if key else None,
        )
        db.add(t)
        db.flush()
        return t.id


def main() -> None:
    print("=" * 72)
    print("PERIODIC MEDUSA SYNC (cron wrapper) HARNESS")
    print("=" * 72)

    with get_session() as db:
        cleanup_by_slug_prefix(db, TEST_SLUG_PREFIX)

    # Three tenants:
    #   A: fully configured, sync should succeed
    #   B: fully configured, sync will raise (isolation test)
    #   C: url-only (no key). Must be filtered out entirely.
    tenant_a = _make_tenant(
        slug=f"{TEST_SLUG_PREFIX}ok-{uuid.uuid4().hex[:6]}",
        url="http://fake-a.test", key="pk_fake_a",
    )
    tenant_b = _make_tenant(
        slug=f"{TEST_SLUG_PREFIX}boom-{uuid.uuid4().hex[:6]}",
        url="http://fake-b.test", key="pk_fake_b",
    )
    tenant_c = _make_tenant(
        slug=f"{TEST_SLUG_PREFIX}nokey-{uuid.uuid4().hex[:6]}",
        url="http://fake-c.test", key=None,
    )
    print(f"seeded: A={tenant_a} B={tenant_b} C={tenant_c}\n")

    try:
        synced: list[str] = []

        async def fake_sync(tenant):
            synced.append(tenant.slug)
            if "boom" in tenant.slug:
                raise RuntimeError("simulated Medusa outage")
            return SyncResult(
                products_seen=3, products_ingested=3, chunks_written=6,
                products_deleted=0, errors=[],
            )

        with patch(
            "app.workers.receptionist.sync_tenant_products",
            side_effect=fake_sync,
        ):
            asyncio.run(sync_all_medusa_tenants({}))

        # ─── CASE 1: filter — tenant C (no key) was skipped ─────────────────
        synced_slugs = sorted(synced)
        c_synced = any("nokey" in s for s in synced)
        record(
            "CASE 1: url-only tenant is filtered out (not synced)",
            not c_synced and len(synced) == 2,
            f"synced slugs={synced_slugs} (want the two fully-configured, "
            f"NOT the -nokey one)",
        )

        # ─── CASE 2: error isolation — A still ran despite B blowing up ─────
        a_synced = any("-ok-" in s for s in synced)
        b_attempted = any("boom" in s for s in synced)
        record(
            "CASE 2: one tenant raising doesn't stop the next tenant",
            a_synced and b_attempted,
            f"A synced={a_synced} B attempted={b_attempted} "
            f"(both want True; the RuntimeError from B must be caught)",
        )

        # ─── CASE 3: empty tick — no matching tenants at all ────────────────
        with get_session() as db:
            cleanup_by_slug_prefix(db, TEST_SLUG_PREFIX)

        empty_synced: list[str] = []

        async def unreachable(tenant):
            empty_synced.append(tenant.slug)
            return SyncResult(0, 0, 0, 0, [])

        with patch(
            "app.workers.receptionist.sync_tenant_products",
            side_effect=unreachable,
        ):
            asyncio.run(sync_all_medusa_tenants({}))

        record(
            "CASE 3: empty tick is a fast no-op",
            empty_synced == [],
            f"attempted syncs={empty_synced} (want [])",
        )

    finally:
        with get_session() as db:
            cleanup_by_slug_prefix(db, TEST_SLUG_PREFIX)

    print("\n" + "=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)
    raise SystemExit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()

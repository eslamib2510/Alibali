"""ARQ worker for processing Chatwoot webhooks asynchronously.

Why this exists
---------------
Chatwoot's Agent Bot webhook expects a fast 200 response (timeout ~5s).
Running the full `handle_message_event` flow (LLM call + DB writes + POST
the reply back to Chatwoot) takes 2-4 seconds end-to-end, which is close
enough to the limit that any cold start or LLM hiccup trips Chatwoot's
"bot errored" handling.

Splitting it: the FastAPI webhook handler now ONLY authenticates the
tenant (fast) and enqueues a job on Redis. The actual work happens in
this worker, which lives in a separate process and isn't on Chatwoot's
clock.

Run the worker
--------------
    arq app.worker.WorkerSettings

In docker-compose it's a second service using the same image with a
different command.

Failure handling
----------------
ARQ retries failed jobs with exponential backoff. Set `max_tries=3` so we
fail loud sooner during v0 — production might bump it higher. Inspect
queued/running/failed jobs from the ARQ CLI (`arq --check`) or a Redis
client (`redis-cli` → `KEYS arq:*`).
"""

from __future__ import annotations

import dataclasses
import logging
from typing import Any
from uuid import UUID

from arq.connections import RedisSettings
from arq.cron import cron
from sqlalchemy import select

from app.config import settings
from app.core.agent import handle_message_event
from app.db.models.tenant import Tenant
from app.db.session import get_session
from app.services.medusa_sync import sync_tenant_products

logger = logging.getLogger(__name__)


async def sync_all_medusa_tenants(ctx: dict) -> None:
    """Cron job: refresh every Medusa-enabled tenant's product catalogue.

    Runs sync_tenant_products for each tenant whose medusa_api_url is set,
    isolating failures per tenant so one broken store doesn't skip the rest.
    Product sync is idempotent (see medusa_sync.py), so re-running on the
    same tenant is safe.
    """
    with get_session() as db:
        # Load only what's needed to detach cleanly. Encrypted key is loaded
        # here (while the session is live) and used later without re-attaching.
        # Require BOTH url and key so partially-configured tenants (a common
        # state during onboarding) don't produce noisy sync errors every tick.
        rows = db.scalars(
            select(Tenant)
            .where(Tenant.medusa_api_url.isnot(None))
            .where(Tenant.medusa_api_key_enc.isnot(None))
        ).all()
        tenants = []
        for t in rows:
            _ = t.medusa_api_url, t.medusa_api_key_enc, t.slug, t.id
            db.expunge(t)
            tenants.append(t)

    if not tenants:
        logger.info("Medusa sync tick: no tenants have medusa_api_url set")
        return

    logger.info("Medusa sync tick: %d tenant(s)", len(tenants))
    for tenant in tenants:
        try:
            result = await sync_tenant_products(tenant)
            logger.info(
                "Medusa sync ok tenant=%s ingested=%d chunks=%d deleted=%d errors=%d",
                tenant.slug,
                result.products_ingested,
                result.chunks_written,
                result.products_deleted,
                len(result.errors),
            )
        except Exception:
            # Swallow so the next tenant still runs. Log with stack for triage.
            logger.exception("Medusa sync failed for tenant=%s", tenant.slug)


async def process_message_event(ctx: dict, payload: dict, tenant_id: str) -> None:
    """ARQ job: run `handle_message_event` in the worker process.

    Args mirror what the webhook handler validated:
      - `payload`: raw Chatwoot webhook body (filtered/validated inside the
        agent loop, not here).
      - `tenant_id`: trusted tenant UUID resolved from the per-tenant
        webhook token at HTTP-handler time. Passed as a string because
        ARQ serializes via msgpack and UUID isn't natively supported.

    Re-raises any exception — ARQ catches it, schedules a retry, and logs
    the failure. We don't need to wrap with our own try/except here.
    """
    # Postgres accepts either a UUID instance or a string for UUID columns,
    # so passing the string straight through to repository.get_tenant_by_id
    # works without conversion. We coerce defensively in case downstream
    # code ever needs a real UUID.
    tid: Any = UUID(tenant_id) if isinstance(tenant_id, str) else tenant_id
    await handle_message_event(payload, tenant_id=tid)


class WorkerSettings:
    """ARQ worker config. Discovered by `arq <this_class>` on startup."""

    functions = [process_message_event, sync_all_medusa_tenants]

    # Every 6 hours (00, 06, 12, 18 UTC). Product catalogues don't move fast
    # enough to justify tighter cadence; anything more urgent should go
    # through live tools (v3 roadmap) instead of the RAG store.
    cron_jobs = [
        cron(sync_all_medusa_tenants, hour={0, 6, 12, 18}, minute=0),
    ]

    # Same widened Redis defaults as the FastAPI side (see lifespan.py) so
    # the worker survives transient blips during burst load instead of
    # crashing the whole process on a cold connection.
    redis_settings = dataclasses.replace(
        RedisSettings.from_dsn(settings.REDIS_URL),
        conn_timeout=10,
        conn_retries=5,
        conn_retry_delay=1,
    )

    # Fail loud sooner during v0 — bump up once we trust the system.
    max_tries = 3

    # Default keep_result is 1 hour; we don't poll for results so set short.
    keep_result = 60  # seconds

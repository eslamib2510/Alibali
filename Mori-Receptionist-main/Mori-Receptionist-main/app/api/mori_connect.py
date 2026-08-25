"""Mori-Connect Agent Bot webhook.

Mori-Connect posts here on every conversation event. We authenticate the
tenant via the per-tenant webhook token in the query string, enqueue the
message for background processing, and return 200 immediately.

Why background: the Agent Bot webhook has a tight timeout (about 5s) and
LLM calls easily push us over it. Returning 200 fast keeps the platform
happy; the actual reply goes back via a separate API call from the worker
process once the LLM finishes.

Configured in the inbox platform at:
  Settings, Integrations, Agent Bot, outgoing URL =
  /api/mori-connect?token=<tenants.webhook_token>

Auth model: each tenant has its own `webhook_token` in the DB. The token is
BOTH authentication ("is this request legit?") AND identity ("which tenant?").
There is no shared secret across tenants: one tenant leaking its token only
exposes that tenant.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query, Request

from app.db import repository
from app.db.session import get_session

logger = logging.getLogger(__name__)

router = APIRouter(redirect_slashes=False)


@router.post("/mori-connect")
async def mori_connect_webhook(
    request: Request,
    token: str = Query(..., description="Per-tenant webhook token (tenants.webhook_token)"),
) -> dict[str, Any]:
    # 1. Auth (fast: single indexed lookup). 401 for any token miss.
    with get_session() as db:
        tenant = repository.get_tenant_by_webhook_token(db, token)
        if tenant is None:
            raise HTTPException(status_code=401, detail="Unauthorized")
        tenant_id = str(tenant.id)  # serialize as string for ARQ/msgpack

    # 2. Parse the body before enqueueing so JSON errors surface here rather
    # than hidden inside a worker run.
    try:
        payload = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON")

    # 3. Enqueue and return immediately. The worker process handles the LLM
    # call + reply post-back.
    #
    # A failure here is the one spot where neither retry mechanism protects
    # us: the job never reaches Redis, so ARQ has nothing to retry, and a 200
    # would tell the platform the delivery succeeded so it won't re-send.
    # Answer 500 instead; the platform retries agent-bot webhooks on 429/500
    # (see Webhooks::Trigger::RETRYABLE_AGENT_BOT_STATUSES upstream), which
    # hands the message back to us instead of dropping it.
    pool = request.app.state.redis_pool
    try:
        await pool.enqueue_job("process_message_event", payload, tenant_id)
    except Exception:
        logger.exception("Failed to enqueue webhook job; asking the platform to retry")
        raise HTTPException(status_code=500, detail="Enqueue failed, retry")

    return {"ok": True}

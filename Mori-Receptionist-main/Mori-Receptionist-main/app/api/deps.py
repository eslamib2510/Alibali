"""Shared FastAPI dependencies.

Currently one: operator-level admin auth for the knowledge endpoints.

Note the difference from the Chatwoot webhook's auth. That token identifies a
tenant — it answers "who is this?" and grants access to that tenant's data
only. This one is an operator key: it can write to ANY tenant's knowledge
base, and the tenant is named in the request body. Very different blast
radius, so it lives in env rather than the database and is never handed to a
tenant.

When tenants get self-serve knowledge management (the v5 admin UI), that needs
a separate per-tenant credential rather than sharing this one.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException

from app.config import settings


def require_admin(authorization: str | None = Header(default=None)) -> None:
    """Gate an endpoint behind the operator admin key.

        @router.post("/knowledge", dependencies=[Depends(require_admin)])

    Expects `Authorization: Bearer <RECEPTIONIST_ADMIN_API_KEY>`.

    Fails closed when the key isn't configured: an unset key means the
    endpoint is unusable, never that it's open. A missing env var in a fresh
    deploy is exactly when an "allow if unset" default would quietly publish
    every tenant's knowledge base to the internet.
    """
    expected = settings.RECEPTIONIST_ADMIN_API_KEY
    if not expected:
        raise HTTPException(
            status_code=503,
            detail="Admin API is not configured (RECEPTIONIST_ADMIN_API_KEY unset)",
        )

    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Unauthorized")

    presented = authorization.removeprefix("Bearer ").strip()

    # compare_digest rather than == : a plain comparison returns as soon as it
    # hits a differing byte, so response time leaks how much of the key was
    # guessed correctly.
    if not secrets.compare_digest(presented, expected):
        raise HTTPException(status_code=401, detail="Unauthorized")

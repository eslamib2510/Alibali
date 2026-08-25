"""Admin endpoints for managing per-tenant knowledge.

Operator-only — every route is behind `require_admin`, and the tenant is named
in the request rather than derived from the caller. See `app/api/deps.py` for
why this key is deliberately not the same kind of credential as a tenant's
webhook token.

Ingestion runs inline rather than on the worker. A FAQ entry is one embedding
call and a handful of inserts, well under a second, and an admin pasting text
into a form wants to know immediately whether it worked. When document upload
lands in v3 and a single source can be hundreds of pages, that one moves to
ARQ on its own queue so it can't starve the message path.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel, Field

from app.api.deps import require_admin
from app.db import repository
from app.db.session import get_session
from app.ingestion import ingestor

logger = logging.getLogger(__name__)

router = APIRouter(dependencies=[Depends(require_admin)])


class KnowledgeIn(BaseModel):
    tenant_id: uuid.UUID
    content: str = Field(min_length=1)
    title: str | None = None
    source_type: str = "faq"
    # Stable external id. Re-posting the same ref replaces that source's
    # chunks instead of adding duplicates — the property that makes a nightly
    # product sync safe to run repeatedly.
    source_ref: str | None = None
    metadata: dict = Field(default_factory=dict)


class KnowledgeOut(BaseModel):
    chunk_ids: list[uuid.UUID]
    chunks_written: int
    chunks_replaced: int


def _require_tenant(tenant_id: uuid.UUID):
    """404 for an unknown tenant rather than letting the insert fail on the
    foreign key — the caller mistyped an id, and that should read as 'no such
    tenant', not as a database error."""
    with get_session() as db:
        tenant = repository.get_tenant_by_id(db, tenant_id)
        if tenant is None:
            raise HTTPException(status_code=404, detail="Tenant not found")


@router.post("/knowledge", response_model=KnowledgeOut, status_code=201)
async def create_knowledge(payload: KnowledgeIn) -> KnowledgeOut:
    """Chunk, embed and store a piece of knowledge for one tenant."""
    _require_tenant(payload.tenant_id)

    try:
        # ingest_text embeds over HTTP and then writes; both block. Off the
        # event loop so a slow embedding call can't stall every other request
        # this process is serving.
        result = await asyncio.to_thread(
            ingestor.ingest_text,
            tenant_id=payload.tenant_id,
            content=payload.content,
            title=payload.title,
            source_type=payload.source_type,
            source_ref=payload.source_ref,
            metadata=payload.metadata,
        )
    except ValueError as e:
        # Bad source_type — the caller's mistake, not ours.
        raise HTTPException(status_code=422, detail=str(e))

    return KnowledgeOut(
        chunk_ids=result.chunk_ids,
        chunks_written=result.chunks_written,
        chunks_replaced=result.chunks_replaced,
    )


@router.get("/knowledge")
async def list_knowledge(
    tenant_id: uuid.UUID = Query(...),
    source_type: str | None = Query(default=None),
    limit: int = Query(default=100, le=500),
) -> dict:
    """List a tenant's chunks. Content is truncated to a preview."""
    _require_tenant(tenant_id)
    items = await asyncio.to_thread(
        ingestor.list_chunks,
        tenant_id=tenant_id,
        source_type=source_type,
        limit=limit,
    )
    return {"items": items, "count": len(items)}


@router.delete("/knowledge/{chunk_id}", status_code=204)
async def delete_knowledge(chunk_id: uuid.UUID, tenant_id: uuid.UUID = Query(...)):
    """Delete one chunk.

    `tenant_id` is required rather than inferred: it scopes the delete so a
    mistyped chunk id can't reach across tenants, and it's what sets the RLS
    session variable.
    """
    deleted = await asyncio.to_thread(
        ingestor.delete_chunk, tenant_id=tenant_id, chunk_id=chunk_id
    )
    if not deleted:
        raise HTTPException(status_code=404, detail="Chunk not found")


@router.delete("/knowledge/source/{source_ref}")
async def delete_knowledge_source(
    source_ref: str, tenant_id: uuid.UUID = Query(...)
) -> dict:
    """Delete every chunk belonging to one source — e.g. a delisted product."""
    deleted = await asyncio.to_thread(
        ingestor.delete_source, tenant_id=tenant_id, source_ref=source_ref
    )
    return {"deleted": deleted}

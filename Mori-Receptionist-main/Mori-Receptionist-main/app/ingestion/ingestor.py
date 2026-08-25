"""Turns raw source material into rows in `knowledge`.

One pipeline: chunk -> embed -> store. Every write path into the knowledge
base goes through here, so chunk sizing, embedding parameters and re-sync
semantics stay in one place rather than being re-invented by the admin
endpoint, the Medusa product sync (v3) and the document uploader.

Re-ingesting the same source replaces its chunks rather than adding more.
Without that, syncing a product catalogue nightly would multiply every
product's chunks by the number of nights, and retrieval would start returning
five copies of a stale description. See `_delete_existing` for why this is a
delete-then-insert instead of a row-by-row upsert.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass

from sqlalchemy import text

from app.db.session import get_session
from app.ingestion.chunker import chunk_text
from app.integrations.gemini import TASK_DOCUMENT, embed_batch

logger = logging.getLogger(__name__)

VALID_SOURCE_TYPES = ("faq", "product", "document")


@dataclass(frozen=True)
class IngestResult:
    """What one ingest call produced."""

    chunk_ids: list[uuid.UUID]
    chunks_written: int
    chunks_replaced: int


def _delete_existing(db, *, tenant_id, source_ref: str) -> int:
    """Remove the chunks a previous ingest of this source left behind.

    Delete-then-insert rather than updating in place: a re-ingest can produce
    a different NUMBER of chunks than last time (the source grew, or the
    chunker's boundaries shifted), so there's no stable per-chunk identity to
    match on. Matching by position would silently pair chunk 3 of the new text
    with chunk 3 of the old and leave orphans past the end.

    Runs in the same transaction as the insert, so a failure mid-ingest rolls
    back to the previous version rather than leaving the tenant with no
    knowledge at all.
    """
    result = db.execute(
        text("""
            DELETE FROM knowledge
            WHERE tenant_id = :tenant_id AND source_ref = :source_ref
        """),
        {"tenant_id": str(tenant_id), "source_ref": source_ref},
    )
    return result.rowcount or 0


def ingest_text(
    *,
    tenant_id,
    content: str,
    title: str | None = None,
    source_type: str = "faq",
    source_ref: str | None = None,
    metadata: dict | None = None,
) -> IngestResult:
    """Chunk, embed and store one piece of source text for a tenant.

    Args:
        tenant_id: owner of the resulting chunks.
        content: raw text. Chunked automatically; pass whole documents.
        title: shown to the model as a heading above the chunk. Carries topic
            information the body often assumes, so it's worth setting.
        source_type: 'faq' | 'product' | 'document'. Enforced by a CHECK
            constraint on the table; validated here to fail with a clear
            message instead of an IntegrityError.
        source_ref: stable external id (e.g. a Medusa product id). When given,
            a previous ingest of the same ref is replaced. When omitted, this
            is always an addition.
        metadata: arbitrary JSON stored alongside each chunk.

    Returns:
        IngestResult with the new chunk ids and how many rows were replaced.

    Blocking: embeds via an HTTP call. Async callers must use
    `asyncio.to_thread`.
    """
    if source_type not in VALID_SOURCE_TYPES:
        raise ValueError(
            f"source_type must be one of {VALID_SOURCE_TYPES}, got {source_type!r}"
        )

    chunks = chunk_text(content)
    if not chunks:
        logger.info("Nothing to ingest for tenant=%s — content was empty", tenant_id)
        return IngestResult(chunk_ids=[], chunks_written=0, chunks_replaced=0)

    # One API call for the whole document rather than one per chunk. Embedding
    # happens BEFORE the transaction opens: it's the slow part, and holding a
    # Postgres connection across it would tie up the pool for no reason.
    vectors = embed_batch(chunks, task_type=TASK_DOCUMENT)

    chunk_ids: list[uuid.UUID] = []
    replaced = 0

    with get_session(tenant_id=tenant_id) as db:
        if source_ref:
            replaced = _delete_existing(db, tenant_id=tenant_id, source_ref=source_ref)

        for chunk, vector in zip(chunks, vectors):
            chunk_id = uuid.uuid4()
            db.execute(
                text("""
                    INSERT INTO knowledge
                        (id, tenant_id, source_type, source_ref, title,
                         content, embedding, metadata)
                    VALUES
                        (:id, :tenant_id, :source_type, :source_ref, :title,
                         :content, CAST(:embedding AS vector),
                         CAST(:metadata AS jsonb))
                """),
                {
                    "id": str(chunk_id),
                    "tenant_id": str(tenant_id),
                    "source_type": source_type,
                    "source_ref": source_ref,
                    "title": title,
                    "content": chunk,
                    "embedding": "[" + ",".join(repr(float(v)) for v in vector) + "]",
                    "metadata": _json(metadata or {}),
                },
            )
            chunk_ids.append(chunk_id)

    logger.info(
        "Ingested %d chunk(s) for tenant=%s source_type=%s source_ref=%s "
        "(replaced %d)",
        len(chunk_ids), tenant_id, source_type, source_ref, replaced,
    )
    return IngestResult(
        chunk_ids=chunk_ids,
        chunks_written=len(chunk_ids),
        chunks_replaced=replaced,
    )


def delete_source(*, tenant_id, source_ref: str) -> int:
    """Remove every chunk belonging to one source. Returns rows deleted."""
    with get_session(tenant_id=tenant_id) as db:
        return _delete_existing(db, tenant_id=tenant_id, source_ref=source_ref)


def delete_chunk(*, tenant_id, chunk_id) -> bool:
    """Remove a single chunk. Returns True if it existed."""
    with get_session(tenant_id=tenant_id) as db:
        result = db.execute(
            text("DELETE FROM knowledge WHERE tenant_id = :t AND id = :id"),
            {"t": str(tenant_id), "id": str(chunk_id)},
        )
        return bool(result.rowcount)


def list_chunks(*, tenant_id, source_type: str | None = None, limit: int = 100):
    """List a tenant's chunks for the admin UI. Content is truncated — the
    caller is rendering a table, not the whole knowledge base."""
    with get_session(tenant_id=tenant_id) as db:
        # The CAST is required, not cosmetic. Postgres can't infer a type for
        # a parameter that only ever appears as `$2 IS NULL`, and psycopg3
        # raises AmbiguousParameter rather than guessing. Naming the type once
        # resolves it for both uses.
        rows = db.execute(
            text("""
                SELECT id::text, source_type, source_ref, title,
                       left(content, 200) AS preview, created_at
                FROM knowledge
                WHERE tenant_id = :tenant_id
                  AND (CAST(:source_type AS text) IS NULL
                       OR source_type = CAST(:source_type AS text))
                ORDER BY created_at DESC
                LIMIT :limit
            """),
            {"tenant_id": str(tenant_id), "source_type": source_type,
             "limit": limit},
        ).mappings().all()
        return [dict(r) for r in rows]


def _json(value: dict) -> str:
    import json

    return json.dumps(value)

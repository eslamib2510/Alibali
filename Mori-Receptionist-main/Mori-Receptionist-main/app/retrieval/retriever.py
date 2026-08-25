"""Hybrid search over per-tenant knowledge chunks.

Runs one Postgres query that combines two independent rankings and fuses them:

  - **Vector similarity** (pgvector HNSW, cosine) catches meaning. "cold
    plunge" finds a chunk about "ice baths" with no shared words.
  - **Full-text search** (tsvector GIN) catches exact tokens. Product codes,
    prices, brand names and rare words are precisely where embeddings are
    weakest — an embedding of "SKU-4471" is close to every other SKU.

Neither alone is good enough for a receptionist, which fields both "do you do
recovery stuff?" and "is the SKU-4471 in stock?" in the same conversation.

Fusion uses Reciprocal Rank Fusion: score = sum over rankers of 1/(k + rank),
with k=60 (the constant from the original RRF paper, and the industry
default). RRF combines *ranks*, not scores, which is what makes it work here —
cosine distance and ts_rank_cd are on incomparable scales, and normalizing
them against each other would mean inventing a conversion with no principled
basis. Rank position sidesteps that entirely.

Tenant isolation holds on two levels: an explicit `tenant_id` predicate in
every CTE, and the RLS policy on `knowledge` reading `app.tenant_id`, which
`get_session(tenant_id=...)` sets. The explicit filter is what the query
planner uses for the index; RLS is what saves us if someone later writes a
query and forgets it.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from sqlalchemy import text
from sqlalchemy.exc import DatabaseError

from app.db.session import get_session
from app.integrations.gemini import TASK_QUERY, embed

logger = logging.getLogger(__name__)

# Over-fetch this many per ranker before fusing. Fusion can only reorder what
# it's given, so each side needs enough depth for a chunk that one ranker
# loves and the other ignores to still surface.
OVER_FETCH = 20

# Chunks handed back to the agent as prompt context. Enough grounding to
# answer from, few enough to leave room for conversation history.
DEFAULT_LIMIT = 5

# RRF smoothing constant. Damps the top ranks so a single ranker's #1 can't
# dominate a chunk that both rankers rate highly.
RRF_K = 60


@dataclass(frozen=True)
class Chunk:
    """One retrieved knowledge chunk."""

    id: str
    title: str | None
    content: str
    source_type: str
    source_ref: str | None
    score: float


# Ranks are assigned AFTER each LIMIT, not before. Windowing over the full
# filtered set and then limiting would make Postgres rank every row in the
# tenant's knowledge base on every query, discarding all but 20 — the index
# would be pointless. The inner subqueries let HNSW and GIN return their top
# rows, and only those get numbered.
_HYBRID_SQL = text("""
WITH vector_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY dist) AS rank
    FROM (
        SELECT id, embedding <=> CAST(:query_vec AS vector) AS dist
        FROM knowledge
        WHERE tenant_id = :tenant_id
        ORDER BY dist
        LIMIT :over_fetch
    ) v
),
fts_hits AS (
    SELECT id, ROW_NUMBER() OVER (ORDER BY score DESC) AS rank
    FROM (
        SELECT k.id, ts_rank_cd(k.content_tsv, q) AS score
        FROM knowledge k, plainto_tsquery('english', :query_text) q
        WHERE k.tenant_id = :tenant_id
          AND k.content_tsv @@ q
        ORDER BY score DESC
        LIMIT :over_fetch
    ) f
),
candidates AS (
    SELECT id FROM vector_hits
    UNION
    SELECT id FROM fts_hits
)
SELECT
    k.id::text          AS id,
    k.title             AS title,
    k.content           AS content,
    k.source_type       AS source_type,
    k.source_ref        AS source_ref,
    COALESCE(1.0 / (:rrf_k + v.rank), 0.0)
      + COALESCE(1.0 / (:rrf_k + f.rank), 0.0) AS score
FROM candidates c
JOIN knowledge k ON k.id = c.id
LEFT JOIN vector_hits v ON v.id = c.id
LEFT JOIN fts_hits   f ON f.id = c.id
ORDER BY score DESC, k.created_at ASC
LIMIT :limit
""")


def _enable_iterative_scan(db) -> None:
    """Let HNSW keep walking when early candidates are filtered out.

    One index spans every tenant's vectors. For a small tenant, the globally
    nearest 20 rows can all belong to somebody else, and the tenant_id filter
    then leaves us with nothing — an empty result for a query that had a
    perfectly good answer. Iterative scan (pgvector 0.8+) makes the index keep
    going until it has enough rows that survive the filter.

    Best-effort: older pgvector doesn't know the setting, and retrieval
    without it is degraded for small tenants but not broken.
    """
    try:
        db.execute(text("SET LOCAL hnsw.iterative_scan = 'relaxed_order'"))
        db.execute(text("SET LOCAL hnsw.max_scan_tuples = 20000"))
    except DatabaseError:
        logger.debug("pgvector iterative scan unavailable; continuing without it")


def search(
    *,
    tenant_id,
    query: str,
    limit: int = DEFAULT_LIMIT,
    over_fetch: int = OVER_FETCH,
) -> list[Chunk]:
    """Return the chunks most relevant to `query`, scoped to one tenant.

    Blocking: embeds the query (an HTTP call) and then hits Postgres. Async
    callers must dispatch through `asyncio.to_thread`, same rule as everything
    in `integrations/gemini.py`.

    Returns an empty list for blank input or when the tenant has no matching
    knowledge — callers should treat "no context" as normal and let the model
    answer without grounding rather than refusing to reply.
    """
    if not query or not query.strip():
        return []

    # RETRIEVAL_QUERY, not RETRIEVAL_DOCUMENT: chunks were embedded as
    # documents, and Gemini places questions and documents in the same space
    # only when each side declares which it is.
    query_vec = embed(query, task_type=TASK_QUERY)
    vec_literal = "[" + ",".join(repr(float(v)) for v in query_vec) + "]"

    with get_session(tenant_id=tenant_id) as db:
        _enable_iterative_scan(db)
        rows = db.execute(
            _HYBRID_SQL,
            {
                "tenant_id": str(tenant_id),
                "query_vec": vec_literal,
                "query_text": query,
                "over_fetch": over_fetch,
                "rrf_k": RRF_K,
                "limit": limit,
            },
        ).mappings().all()

    return [
        Chunk(
            id=row["id"],
            title=row["title"],
            content=row["content"],
            source_type=row["source_type"],
            source_ref=row["source_ref"],
            score=float(row["score"]),
        )
        for row in rows
    ]


def format_context(chunks: list[Chunk]) -> str:
    """Render retrieved chunks as a prompt block.

    Titles are included when present because they carry topic information the
    chunk body often assumes ("Cancellations" above text that never repeats
    the word). Empty string for no chunks, so the caller can drop the whole
    section rather than paste an empty heading into the prompt.
    """
    if not chunks:
        return ""

    parts = []
    for chunk in chunks:
        header = f"[{chunk.title}]\n" if chunk.title else ""
        parts.append(f"{header}{chunk.content}")
    return "\n---\n".join(parts)

"""knowledge table with pgvector + tsvector + RLS

Per-tenant knowledge chunks for RAG. Sources include FAQ text (admin
inserted), Medusa product descriptions (v3), and later uploaded documents.

Design notes:
- One table for all tenants. `tenant_id` filter + RLS enforces isolation.
- `content_tsv` is a GENERATED column so full-text search is always in sync
  with `content`. No trigger, no application code.
- `embedding vector(1536)` matches `gemini-embedding-001` truncated via
  Matryoshka Representation Learning (no quality loss vs 3072 default).
- HNSW index on embedding, GIN index on tsvector. Hybrid search combines
  both via RRF at query time (see app/retrieval/retriever.py).
- pgvector 0.8+ iterative scan is set per-session in app/db/session.py so
  small tenants still get good recall when candidates are filtered out.

Revision ID: 3e5b8a2c7d10
Revises: 2c9a1d4e7f53
Create Date: 2026-07-25 00:01:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


revision: str = "3e5b8a2c7d10"
down_revision: Union[str, Sequence[str], None] = "2c9a1d4e7f53"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # 1. pgvector extension. Idempotent — Neon has this available.
    op.execute("CREATE EXTENSION IF NOT EXISTS vector")

    # 2. Table. `embedding` and `content_tsv` use raw SQL because SQLAlchemy
    #    core doesn't have a first-class vector type without the pgvector
    #    Python package as a dependency of the migration itself; simpler to
    #    keep the migration self-contained.
    op.execute("""
        CREATE TABLE knowledge (
            id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
            tenant_id      UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
            source_type    TEXT NOT NULL CHECK (source_type IN ('faq', 'product', 'document')),
            source_ref     TEXT,
            title          TEXT,
            content        TEXT NOT NULL,
            content_tsv    tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
            embedding      vector(1536) NOT NULL,
            metadata       JSONB NOT NULL DEFAULT '{}'::jsonb,
            created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
            updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
        )
    """)

    # 3. HNSW index for vector similarity. m=16, ef_construction=64 are the
    #    pgvector defaults — good balance of recall + build cost at our scale.
    op.execute("""
        CREATE INDEX ix_knowledge_embedding
            ON knowledge USING hnsw (embedding vector_cosine_ops)
            WITH (m = 16, ef_construction = 64)
    """)

    # 4. GIN index for the lexical half of hybrid search.
    op.execute("""
        CREATE INDEX ix_knowledge_tsv
            ON knowledge USING gin (content_tsv)
    """)

    # 5. Tenant + source_type composite for admin listings and per-tenant scans
    #    when RLS is bypassed (e.g. superuser dashboards later).
    op.execute("""
        CREATE INDEX ix_knowledge_tenant_source
            ON knowledge (tenant_id, source_type)
    """)

    # 6. Row-Level Security. Every query on `knowledge` is auto-filtered by
    #    the tenant_id set in the session via SET LOCAL app.tenant_id = '...'
    #    (done in app/db/session.py for each get_session()). Defense-in-depth:
    #    even if a query forgets WHERE tenant_id, RLS blocks cross-tenant reads.
    op.execute("ALTER TABLE knowledge ENABLE ROW LEVEL SECURITY")
    op.execute("""
        CREATE POLICY tenant_isolation ON knowledge
            USING (tenant_id = NULLIF(current_setting('app.tenant_id', true), '')::uuid)
    """)


def downgrade() -> None:
    op.execute("DROP POLICY IF EXISTS tenant_isolation ON knowledge")
    op.execute("ALTER TABLE knowledge DISABLE ROW LEVEL SECURITY")
    op.execute("DROP INDEX IF EXISTS ix_knowledge_tenant_source")
    op.execute("DROP INDEX IF EXISTS ix_knowledge_tsv")
    op.execute("DROP INDEX IF EXISTS ix_knowledge_embedding")
    op.execute("DROP TABLE IF EXISTS knowledge")
    # Do NOT drop the vector extension — other things may use it later.

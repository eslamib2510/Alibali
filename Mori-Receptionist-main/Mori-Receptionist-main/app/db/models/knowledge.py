"""Knowledge — one chunk of retrievable content per tenant.

Powers the RAG layer. Sources:
- `faq`      admin-inserted FAQ text
- `product`  Medusa product descriptions (v3)
- `document` uploaded docs (later)

The `embedding` column uses pgvector (1536 dims to match gemini-embedding-001
truncated via Matryoshka Representation Learning — no quality loss vs 3072).

`content_tsv` is a Postgres GENERATED column populated by the migration
(`content_tsv tsvector GENERATED ALWAYS AS (to_tsvector('english', content))
STORED`). Declared here as a Computed column so SQLAlchemy knows it's
DB-side and won't try to insert into it.

Tenant isolation is enforced at two layers:
- Explicit `WHERE tenant_id` in every query (via repository funcs)
- Postgres RLS policy `tenant_isolation` reading `app.tenant_id` session var
  (see the 2026_07_25 migration)
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import TYPE_CHECKING

from pgvector.sqlalchemy import Vector
from sqlalchemy import CheckConstraint, Computed, DateTime, ForeignKey, Index, Text
from sqlalchemy.dialects.postgresql import JSONB, TSVECTOR, UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db.base import Base, utcnow

if TYPE_CHECKING:
    from app.db.models.tenant import Tenant


class Knowledge(Base):
    __tablename__ = "knowledge"

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True), primary_key=True, default=uuid.uuid4
    )
    tenant_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("tenants.id", ondelete="CASCADE"),
        nullable=False,
    )

    # 'faq' | 'product' | 'document' — enforced by CHECK constraint at DB level.
    source_type: Mapped[str] = mapped_column(Text, nullable=False)
    # Optional external ID (e.g. Medusa product id) so re-syncs can update
    # instead of duplicating.
    source_ref: Mapped[str | None] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    content: Mapped[str] = mapped_column(Text, nullable=False)

    # Postgres computes this from `content`; SQLAlchemy must not write to it.
    content_tsv: Mapped[str | None] = mapped_column(
        TSVECTOR,
        Computed("to_tsvector('english', content)", persisted=True),
    )

    # gemini-embedding-001 output truncated to 1536 dims (Matryoshka).
    # HNSW index defined in migration, not here.
    embedding: Mapped[list[float]] = mapped_column(Vector(1536), nullable=False)

    metadata_: Mapped[dict] = mapped_column("metadata", JSONB, default=dict, nullable=False)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utcnow, onupdate=utcnow, nullable=False
    )

    tenant: Mapped["Tenant"] = relationship(lazy="raise")

    __table_args__ = (
        CheckConstraint(
            "source_type IN ('faq', 'product', 'document')",
            name="knowledge_source_type_check",
        ),
        # Composite for admin listings ("all FAQs for tenant X").
        Index("ix_knowledge_tenant_source", "tenant_id", "source_type"),
        # HNSW on embedding — declared here so autogenerate sees it and
        # doesn't propose dropping. Actual CREATE INDEX with vector_cosine_ops
        # lives in the migration.
        Index(
            "ix_knowledge_embedding",
            "embedding",
            postgresql_using="hnsw",
            postgresql_with={"m": 16, "ef_construction": 64},
            postgresql_ops={"embedding": "vector_cosine_ops"},
        ),
        # GIN on the generated tsvector for lexical search.
        Index("ix_knowledge_tsv", "content_tsv", postgresql_using="gin"),
    )

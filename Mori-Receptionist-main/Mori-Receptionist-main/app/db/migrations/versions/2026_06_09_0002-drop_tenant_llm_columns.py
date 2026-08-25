"""drop tenant llm_provider + llm_model columns

The model is operator-wide via settings.RECEPTIONIST_DEFAULT_MODEL, not
per-tenant. Per-tenant override can be re-added later (one new column,
nullable, falls back to the env default) the day premium tiers or A/B
testing actually need it.

Revision ID: 8a1c4d6f2b93
Revises: 5f3a2e1b9c47
Create Date: 2026-06-09 00:02:00.000000+00:00
"""
from __future__ import annotations

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = "8a1c4d6f2b93"
down_revision: Union[str, Sequence[str], None] = "5f3a2e1b9c47"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    op.drop_column("tenants", "llm_model")
    op.drop_column("tenants", "llm_provider")


def downgrade() -> None:
    op.add_column(
        "tenants",
        sa.Column(
            "llm_provider",
            sa.String(length=32),
            nullable=False,
            server_default="gemini",
        ),
    )
    op.add_column(
        "tenants",
        sa.Column(
            "llm_model",
            sa.String(length=64),
            nullable=False,
            server_default="gemini-2.0-flash",
        ),
    )

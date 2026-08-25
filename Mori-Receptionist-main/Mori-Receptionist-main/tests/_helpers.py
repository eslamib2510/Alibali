"""Shared plumbing for proof_of_* harnesses.

Every harness needs the same shape of "delete only rows this run created,
never a live tenant". Extracted here so the six copies stay in sync when
a new dependent table (like `knowledge`) gets added and needs cascading.
"""

from __future__ import annotations

from sqlalchemy import text


def cleanup_by_mori_connect_account_id(db, account_id: int) -> None:
    """Delete tenants (and their dependent rows) matching this account_id.

    Harnesses use a NEGATIVE account_id sentinel so this can never touch a
    real tenant (real inbox platform account ids are positive).
    """
    _cascade_delete(
        db,
        tenant_id_select=(
            "SELECT id FROM tenants WHERE mori_connect_account_id = :v"
        ),
        params={"v": account_id},
    )


def cleanup_by_slug_prefix(db, prefix: str) -> None:
    """Delete tenants (and their dependent rows) whose slug starts with prefix.

    Harnesses that create multiple tenants per run (proof_of_rls, ingest,
    retriever, medusa_sync) tag each slug with a per-harness prefix so
    cleanup here scopes to just that harness's rows.
    """
    _cascade_delete(
        db,
        tenant_id_select="SELECT id FROM tenants WHERE slug LIKE :v",
        params={"v": f"{prefix}%"},
    )


def _cascade_delete(db, *, tenant_id_select: str, params: dict) -> None:
    """Delete tenants matching a select + everything they own.

    Manual cascade rather than trusting ON DELETE CASCADE so the order is
    explicit and safe against future migrations changing cascade rules.
    """
    ids = [row[0] for row in db.execute(text(tenant_id_select), params).all()]
    if not ids:
        return
    db.execute(text("DELETE FROM knowledge WHERE tenant_id = ANY(:ids)"),
               {"ids": ids})
    db.execute(text("DELETE FROM messages WHERE tenant_id = ANY(:ids)"),
               {"ids": ids})
    db.execute(text("DELETE FROM conversations WHERE tenant_id = ANY(:ids)"),
               {"ids": ids})
    db.execute(text("DELETE FROM customers WHERE tenant_id = ANY(:ids)"),
               {"ids": ids})
    db.execute(text("DELETE FROM tenants WHERE id = ANY(:ids)"),
               {"ids": ids})

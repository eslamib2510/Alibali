"""Live, tenant-scoped Medusa tools for stock and price questions.

Catalogue text in RAG is useful to identify a product, but it must not be
treated as the source of truth for values that can change while a customer is
chatting.  These tools fetch a single published product from the tenant's
Store API immediately before the agent replies.
"""

from __future__ import annotations

import logging
from typing import Any

from langchain_core.tools import BaseTool, tool
from sqlalchemy import select

from app.core import crypto
from app.db.models.tenant import Tenant
from app.db.session import get_session
from app.integrations.medusa import MedusaClient

logger = logging.getLogger(__name__)


def _price(variant: dict[str, Any]) -> str | None:
    calculated = variant.get("calculated_price") or {}
    amount = calculated.get("calculated_amount")
    if amount is None:
        return None
    currency = (calculated.get("currency_code") or "").upper()
    return f"{amount:,.0f} {currency}".strip()


def _stock(variant: dict[str, Any]) -> str:
    """Describe Store API inventory without guessing when it is unavailable."""
    if variant.get("manage_inventory") is False:
        return "available (inventory is not tracked)"
    quantity = variant.get("inventory_quantity")
    if quantity is None:
        return "availability unavailable"
    if quantity > 0:
        return f"in stock ({quantity} available)"
    if variant.get("allow_backorder"):
        return "available on backorder"
    return "out of stock"


def _variant_name(variant: dict[str, Any]) -> str:
    return (variant.get("title") or variant.get("sku") or variant.get("id") or "variant").strip()


def build_live_catalog_tools(tenant_id: str) -> list[BaseTool]:
    """Return live tools for a configured tenant, otherwise no tools.

    The tenant identifier is used only while constructing closures.  It is
    never exposed in either tool schema, preventing an LLM instruction from
    redirecting a Store API call to another tenant's credentials.
    """
    with get_session() as db:
        tenant = db.scalar(select(Tenant).where(Tenant.id == tenant_id))
        if tenant is None or not tenant.medusa_api_url:
            return []
        medusa_api_url = tenant.medusa_api_url
        try:
            api_key = crypto.get_medusa_key(tenant)
        except crypto.EncryptionError:
            logger.exception(
                "Live catalogue disabled: unreadable Medusa key for tenant=%s", tenant_id
            )
            return []

    if not api_key:
        return []
    client = MedusaClient(base_url=medusa_api_url, api_key=api_key)

    async def fetch(product_id: str) -> dict[str, Any] | None:
        regions = await client.list_regions()
        region_id = regions[0].get("id") if regions else None
        return await client.get_product(product_id, region_id=region_id)

    @tool
    async def get_product_stock(product_id: str) -> str:
        """Get real-time stock for every variant of one store product.

        Use this for availability questions only after identifying the exact
        Medusa Product ID with search_knowledge. Never infer stock from old
        conversation text or a RAG result.

        Args:
            product_id: Exact Medusa Product ID from search_knowledge.
        """
        product = await fetch(product_id)
        if not product:
            return f"No currently published product was found for ID {product_id!r}."
        variants = product.get("variants") or []
        if not variants:
            return f"{product.get('title') or product_id}: no variants returned by the store."
        lines = [f"Live stock for {product.get('title') or product_id}:"]
        lines.extend(f"- {_variant_name(v)}: {_stock(v)}" for v in variants)
        return "\n".join(lines)

    @tool
    async def get_product_price(product_id: str) -> str:
        """Get the current regional price for every variant of one store product.

        Use this for price questions only after identifying the exact Medusa
        Product ID with search_knowledge. The returned price is live and is
        preferred over a price that appears in older knowledge-base text.

        Args:
            product_id: Exact Medusa Product ID from search_knowledge.
        """
        product = await fetch(product_id)
        if not product:
            return f"No currently published product was found for ID {product_id!r}."
        variants = product.get("variants") or []
        if not variants:
            return f"{product.get('title') or product_id}: no variants returned by the store."
        lines = [f"Live price for {product.get('title') or product_id}:"]
        for variant in variants:
            lines.append(f"- {_variant_name(variant)}: {_price(variant) or 'price unavailable'}")
        return "\n".join(lines)

    return [get_product_stock, get_product_price]

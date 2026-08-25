"""Medusa v2 Store API client.

Thin wrapper around the handful of `/store` endpoints the receptionist needs
to sync a tenant's catalogue into `knowledge`. Deliberately not a full SDK
port — we consume raw dicts and let the sync layer decide which fields matter,
so a schema change in Medusa doesn't ripple through a bunch of pydantic models
we have to keep in step.

Authentication is the customer-facing PUBLISHABLE key (`pk_...`), sent as
`x-publishable-api-key`. That's the same key Mori-Store's storefront uses (see
apps/storefront/src/lib/config.ts), and it returns only published products in
the tenant's default sales channel — the correct scope for a receptionist that
answers customer questions.

An admin/secret key would work for `/admin/products` but would also expose
drafts and unpublished data, which we neither want nor need.
"""

from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# 20s matches the Chatwoot client. Medusa's `/store/products` with heavy
# field selection can be slow on cold caches; anything past this window
# probably means the tenant's Medusa is down and we should surface that
# rather than hang the sync job.
DEFAULT_TIMEOUT = 20.0

# Field selection asks Medusa to expand relations we need for the embedded
# text blob. Matches the storefront's own query (minus inventory_quantity —
# stock is deliberately kept out of the RAG store; see medusa_sync.py).
# Without this, `calculated_price` on variants comes back undefined.
DEFAULT_FIELDS = (
    "*variants.calculated_price,"
    "*variants.options,"
    "+metadata,"
    "+tags,"
    "+categories,"
    "+collection,"
    "*images"
)

# Store endpoints omit these inventory-management fields unless explicitly
# requested with `+`. Keeping the selection separate from DEFAULT_FIELDS
# prevents catalogue sync from accidentally embedding volatile inventory.
LIVE_PRODUCT_FIELDS = (
    DEFAULT_FIELDS
    + ",+variants.inventory_quantity,+variants.manage_inventory,+variants.allow_backorder"
)


class MedusaClient:
    """Store-API-scoped client for one tenant's Medusa backend."""

    def __init__(
        self,
        base_url: str,
        api_key: str,
        *,
        timeout: float = DEFAULT_TIMEOUT,
    ):
        if not base_url:
            raise ValueError("MedusaClient requires a base_url")
        if not api_key:
            raise ValueError("MedusaClient requires a publishable api_key")
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.timeout = timeout

    def _headers(self) -> dict[str, str]:
        # x-publishable-api-key is the Medusa v2 store-side auth header.
        # Content-Type isn't strictly required for GETs but keeps parity with
        # other integrations in the codebase.
        return {
            "x-publishable-api-key": self.api_key,
            "Content-Type": "application/json",
        }

    async def list_regions(self) -> list[dict[str, Any]]:
        """Fetch this store's regions.

        Product prices are region-scoped in Medusa — a variant's
        `calculated_price` only populates when the request declares which
        region the customer is in. We pick the first region and thread its id
        through `list_products` so embedded prices reflect a real regional
        price rather than being blank.
        """
        url = f"{self.base_url}/store/regions"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(url, headers=self._headers())
        if r.status_code >= 400:
            logger.error("Medusa list_regions failed: HTTP %d %s", r.status_code, r.text[:300])
            r.raise_for_status()
        body = r.json()
        # v2 returns {"regions": [...]}; be defensive about a bare list too.
        if isinstance(body, dict) and "regions" in body:
            return list(body["regions"])
        return list(body)

    async def list_products(
        self,
        *,
        limit: int = 100,
        offset: int = 0,
        region_id: str | None = None,
        fields: str = DEFAULT_FIELDS,
    ) -> tuple[list[dict[str, Any]], int]:
        """One page of `/store/products`. Returns `(products, total_count)`.

        `total_count` is the total available in the store, not the length of
        this page — the sync loop uses it to decide when to stop paginating.
        `region_id` is optional but strongly recommended: without it, variant
        `calculated_price` fields come back missing.
        """
        params: dict[str, Any] = {"limit": limit, "offset": offset, "fields": fields}
        if region_id:
            params["region_id"] = region_id

        url = f"{self.base_url}/store/products"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            r = await client.get(url, headers=self._headers(), params=params)

        if r.status_code >= 400:
            logger.error("Medusa list_products failed: HTTP %d %s", r.status_code, r.text[:300])
            r.raise_for_status()

        body = r.json()
        products = list(body.get("products") or [])
        count = int(body.get("count") or len(products))
        return products, count

    async def get_product(
        self,
        product_id: str,
        *,
        region_id: str | None = None,
        fields: str = LIVE_PRODUCT_FIELDS,
    ) -> dict[str, Any] | None:
        """Fetch one published product from the Store API.

        This is deliberately separate from catalogue sync: a customer-facing
        question about a specific product needs the value Medusa has *now*,
        not the value that was embedded several hours ago.  A missing product
        is returned as ``None`` so agent tools can give the model a useful
        result instead of turning a discontinued product into a failed turn.
        """
        params: dict[str, Any] = {"fields": fields}
        if region_id:
            params["region_id"] = region_id
        url = f"{self.base_url}/store/products/{product_id}"
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            response = await client.get(url, headers=self._headers(), params=params)
        if response.status_code == 404:
            return None
        if response.status_code >= 400:
            logger.error(
                "Medusa get_product failed: HTTP %d %s",
                response.status_code,
                response.text[:300],
            )
            response.raise_for_status()
        body = response.json()
        # Medusa v2 returns {"product": {...}}. Supporting a bare object
        # makes this client tolerant of compatible storefront proxies.
        return body.get("product") if isinstance(body, dict) and "product" in body else body

"""Live Medusa stock/price tool harness.

    python -m tests.proof_of_live_catalog

No database or HTTP server is needed: the tenant lookup and Medusa client are
replaced at their boundary, while the actual LangChain tools are invoked.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace
from unittest.mock import patch

from app.integrations.medusa import MedusaClient
from app.tools.live_catalog import build_live_catalog_tools


class FakeSession:
    def __init__(self, tenant):
        self.tenant = tenant

    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return False

    def scalar(self, _query):
        return self.tenant


class FakeMedusaClient:
    product = {
        "title": "Recovery Bath",
        "variants": [
            {
                "id": "var_black",
                "title": "Black",
                "inventory_quantity": 3,
                "manage_inventory": True,
                "allow_backorder": False,
                "calculated_price": {"calculated_amount": 1_250_000, "currency_code": "idr"},
            },
            {
                "id": "var_white",
                "title": "White",
                "inventory_quantity": 0,
                "manage_inventory": True,
                "allow_backorder": True,
                "calculated_price": {"calculated_amount": 1_300_000, "currency_code": "idr"},
            },
        ],
    }
    calls: list[tuple[str, str | None]] = []

    def __init__(self, **_kwargs):
        pass

    async def list_regions(self):
        return [{"id": "reg_idr"}]

    async def get_product(self, product_id, *, region_id=None):
        self.calls.append((product_id, region_id))
        return self.product if product_id == "prod_bath" else None


class FakeResponse:
    def __init__(self, status_code, body):
        self.status_code = status_code
        self._body = body
        self.text = str(body)

    def json(self):
        return self._body

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeHttpClient:
    responses: list[FakeResponse] = []
    requests: list[dict] = []

    def __init__(self, **_kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *_args):
        return False

    async def get(self, url, *, headers, params):
        self.requests.append({"url": url, "headers": headers, "params": params})
        return self.responses.pop(0)


async def main() -> None:
    tenant = SimpleNamespace(medusa_api_url="https://store.test")
    with (
        patch("app.tools.live_catalog.get_session", lambda: FakeSession(tenant)),
        patch("app.tools.live_catalog.crypto.get_medusa_key", lambda _tenant: "pk_test"),
        patch("app.tools.live_catalog.MedusaClient", FakeMedusaClient),
    ):
        tools = {tool.name: tool for tool in build_live_catalog_tools("tenant-a")}
        stock = await tools["get_product_stock"].ainvoke({"product_id": "prod_bath"})
        price = await tools["get_product_price"].ainvoke({"product_id": "prod_bath"})
        missing = await tools["get_product_stock"].ainvoke({"product_id": "prod_missing"})

    FakeHttpClient.requests = []
    FakeHttpClient.responses = [
        FakeResponse(200, {"product": {"id": "prod_bath"}}),
        FakeResponse(404, {"message": "not found"}),
    ]
    with patch("app.integrations.medusa.httpx.AsyncClient", FakeHttpClient):
        client = MedusaClient("https://store.test/", "pk_test")
        fetched = await client.get_product("prod_bath", region_id="reg_idr")
        not_found = await client.get_product("prod_missing")
    request = FakeHttpClient.requests[0]

    checks = [
        ("stock reports live quantity", "Black: in stock (3 available)" in stock),
        ("stock reports backorder", "White: available on backorder" in stock),
        ("price reports live regional amounts", "Black: 1,250,000 IDR" in price),
        ("missing product is explicit", "No currently published product" in missing),
        (
            "each lookup uses the store region",
            all(region == "reg_idr" for _, region in FakeMedusaClient.calls),
        ),
        (
            "client requests live calculated price and inventory fields",
            fetched == {"id": "prod_bath"}
            and request["headers"]["x-publishable-api-key"] == "pk_test"
            and request["params"]["region_id"] == "reg_idr"
            and "+variants.inventory_quantity" in request["params"]["fields"],
        ),
        ("client treats a Store API 404 as product unavailable", not_found is None),
    ]
    for name, passed in checks:
        print(f"{'PASS' if passed else 'FAIL'}  {name}")
    raise SystemExit(0 if all(passed for _, passed in checks) else 1)


if __name__ == "__main__":
    asyncio.run(main())

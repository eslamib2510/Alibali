"""API router aggregator. Include new sub-routers here."""

from __future__ import annotations

from fastapi import APIRouter

from app.api import health, knowledge, mori_connect

api_router = APIRouter()
api_router.include_router(health.router, tags=["health"])
api_router.include_router(mori_connect.router, tags=["mori-connect"])
api_router.include_router(knowledge.router, tags=["knowledge"])

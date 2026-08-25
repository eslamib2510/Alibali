"""Liveness + readiness."""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {"ok": True}

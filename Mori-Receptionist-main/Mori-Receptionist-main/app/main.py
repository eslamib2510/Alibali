"""FastAPI entrypoint for Mori-Receptionist.

Two responsibilities on startup:
  1. Configure logging.
  2. Open an ARQ Redis pool and stash it on `app.state` so webhook routes can
     enqueue background work without opening a new pool per request.

Routes are mounted under `/api/`. No version prefix yet — add one only when
we need to run multiple versions side by side.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from arq import create_pool
from arq.connections import RedisSettings
from fastapi import FastAPI

from app.api.router import api_router
from app.config import settings
from app.logging import configure_logging


@asynccontextmanager
async def lifespan(app: FastAPI):
    configure_logging()

    # Widened retry settings — Redis is transient by nature and Neon/Upstash
    # can nap between requests. Better to retry than crash the whole process.
    redis_settings = RedisSettings.from_dsn(settings.REDIS_URL)
    redis_settings.conn_timeout = 10
    redis_settings.conn_retries = 5
    redis_settings.conn_retry_delay = 1

    app.state.redis_pool = await create_pool(redis_settings)
    try:
        yield
    finally:
        await app.state.redis_pool.close()


app = FastAPI(
    title="Mori Receptionist",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(api_router, prefix="/api")


@app.get("/")
async def root() -> dict:
    return {"service": "mori-receptionist", "version": "0.1.0"}

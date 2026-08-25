"""Settings for Mori-Receptionist.

Reads from .env via pydantic-settings. Every knob the code needs is declared
here so a missing env var fails loud at startup, not deep in a request.
"""

from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # ─── App ────────────────────────────────────────────────────────────────
    APP_ENV: str = "development"
    LOG_LEVEL: str = "INFO"

    # ─── Database (Neon) ────────────────────────────────────────────────────
    RECEPTIONIST_DATABASE_URL: str
    RECEPTIONIST_DATABASE_DIRECT_URL: str | None = None

    # ─── Redis / ARQ ────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ─── Encryption ─────────────────────────────────────────────────────────
    RECEPTIONIST_ENCRYPTION_KEY: str | None = None

    # ─── Admin API ──────────────────────────────────────────────────────────
    # Operator key for the knowledge endpoints. Unlike a tenant's
    # `webhook_token`, this grants write access to EVERY tenant's knowledge
    # base, so it stays in env and is never handed to a tenant. Unset means
    # the admin API refuses all requests (503), never that it's open.
    RECEPTIONIST_ADMIN_API_KEY: str | None = None

    # ─── LLM ────────────────────────────────────────────────────────────────
    GEMINI_API_KEY: str | None = None
    RECEPTIONIST_DEFAULT_MODEL: str = "gemini-3.1-flash-lite"

    # ─── Mori-Connect (inbox platform) ──────────────────────────────────────
    # Base URL of the inbox platform we integrate with. Defaults to Chatwoot
    # Cloud since Mori-Connect ships wire-compatible with upstream Chatwoot.
    # Point at a self-hosted instance if the tenant runs their own.
    MORI_CONNECT_BASE_URL: str = "https://app.chatwoot.com"

    # Public URL where the inbox platform can reach this service. Used by
    # manage_tenant.py to build the Agent Bot's outgoing_url when
    # --bot-endpoint isn't passed explicitly. Unset is fine for local dev;
    # the script tells you and you can either set this or pass --bot-endpoint.
    RECEPTIONIST_PUBLIC_URL: str | None = None


settings = Settings()

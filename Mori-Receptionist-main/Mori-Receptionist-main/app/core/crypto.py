"""Symmetric encryption for tenant secrets at rest.

Tenant rows hold third-party API tokens (Mori-Connect, Medusa). Those columns
are stored encrypted with Fernet so a DB credential leak doesn't immediately
expose every tenant's downstream credentials.

Setup
-----
Generate a key once and put it in env as `RECEPTIONIST_ENCRYPTION_KEY`:

    python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"

The key never changes after that. Rotating it is a separate ceremony (decrypt
with old key, re-encrypt with new key, store new key), not implemented yet.

Helpers
-------
- `encrypt(plaintext)` / `decrypt(ciphertext)`: raw primitives.
- `get_mori_connect_token(tenant)` / `get_medusa_key(tenant)`: convenience
  readers that hide the decryption step. Most callers should use these.
"""

from __future__ import annotations

from functools import lru_cache
from typing import TYPE_CHECKING

from cryptography.fernet import Fernet, InvalidToken

from app.config import settings

if TYPE_CHECKING:
    from app.db.models.tenant import Tenant


class EncryptionError(RuntimeError):
    """Raised on missing/invalid key or unreadable ciphertext."""


@lru_cache(maxsize=1)
def _fernet() -> Fernet:
    """Build the Fernet instance once. Cached so we don't re-parse the key
    on every encrypt/decrypt call.

    Raises EncryptionError early if the key isn't configured — better to
    surface this at startup of any flow that needs it than to fail deep
    inside a webhook handler.
    """
    key = settings.RECEPTIONIST_ENCRYPTION_KEY
    if not key:
        raise EncryptionError(
            "RECEPTIONIST_ENCRYPTION_KEY is not set. Generate one with:\n"
            "  python -c \"from cryptography.fernet import Fernet; "
            "print(Fernet.generate_key().decode())\""
        )
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError) as e:
        raise EncryptionError(f"Invalid RECEPTIONIST_ENCRYPTION_KEY: {e}") from e


def encrypt(plaintext: str) -> str:
    """Encrypt a plaintext string. Returns ciphertext as a UTF-8 string
    safe to store in a TEXT column."""
    return _fernet().encrypt(plaintext.encode("utf-8")).decode("utf-8")


def encrypt_optional(plaintext: str | None) -> str | None:
    """Same as `encrypt()` but passes through None / empty strings. Use when
    persisting optional secret columns (medusa_api_key_enc, etc.) so the row
    holds NULL rather than an encrypted empty string."""
    if not plaintext:
        return None
    return encrypt(plaintext)


def decrypt(ciphertext: str) -> str:
    """Decrypt a stored ciphertext back to plaintext. Raises EncryptionError
    if the ciphertext is corrupted or was encrypted with a different key."""
    try:
        return _fernet().decrypt(ciphertext.encode("utf-8")).decode("utf-8")
    except InvalidToken as e:
        raise EncryptionError(
            "Failed to decrypt — wrong key or tampered ciphertext"
        ) from e


# ─── Convenience accessors ──────────────────────────────────────────────────


def get_mori_connect_token(tenant: "Tenant") -> str:
    """Decrypt and return the tenant's Mori-Connect user API token.

    This is the operator's user_access_token, used for admin operations
    like provisioning custom attributes, listing inboxes, creating the bot,
    etc. For posting messages AS the bot, use `get_mori_connect_bot_token()`
    instead so the platform stamps the message with sender.type='agent_bot'.
    """
    if not tenant.mori_connect_api_token_enc:
        raise EncryptionError(
            f"Tenant {tenant.slug} has no Mori-Connect API token configured."
        )
    return decrypt(tenant.mori_connect_api_token_enc)


def get_mori_connect_bot_token(tenant: "Tenant") -> str:
    """Decrypt and return the bot's own Mori-Connect api_access_token.

    Captured at agent-bot creation time (see scripts/manage_tenant.py) and
    used at runtime to post replies. Posting with this token makes the
    outgoing message echo come back with sender.type='agent_bot' so our
    human-takeover detector doesn't misfire on the bot's own messages.
    """
    if not tenant.mori_connect_bot_token_enc:
        raise EncryptionError(
            f"Tenant {tenant.slug} has no Mori-Connect bot token configured. "
            "Re-run manage_tenant.py to capture it from the platform."
        )
    return decrypt(tenant.mori_connect_bot_token_enc)


def get_medusa_key(tenant: "Tenant") -> str | None:
    """Decrypt and return the tenant's Medusa API key, or None if unset."""
    if not tenant.medusa_api_key_enc:
        return None
    return decrypt(tenant.medusa_api_key_enc)

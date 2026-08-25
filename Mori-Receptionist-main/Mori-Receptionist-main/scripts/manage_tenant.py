"""Create or update a Receptionist tenant (upsert by slug).

One script, one mental model: "this is the desired state for tenant <slug>".
If the slug doesn't exist, the tenant is created. If it does, only the fields
you pass on this invocation are updated; everything else stays as-is.

Examples
--------
Create a new tenant (full minimum):
    python -m scripts.manage_tenant \\
        --slug alabali --name "AlaBali" \\
        --mori-connect-account-id 123 \\
        --mori-connect-api-token <user_access_token> \\
        --prompt-file prompts/alabali.txt

Create a draft tenant (token can be added later):
    python -m scripts.manage_tenant --slug acme --name "Acme" --mori-connect-account-id 456

Patch one field on an existing tenant:
    python -m scripts.manage_tenant --slug alabali --mori-connect-api-token cw_new_xxx
    python -m scripts.manage_tenant --slug alabali --prompt-file prompts/alabali_v2.txt
    python -m scripts.manage_tenant --slug alabali --no-notify-admin-on-message
    python -m scripts.manage_tenant --slug <slug> \\
        --medusa-api-url https://<their-medusa-host> --medusa-api-key pk_...

What it does
------------
CREATE path (slug not found):
    1. Validate the inbox platform token + account_id (if a token was given)
    2. Insert the Tenant row
    3. Provision the `bot_mode` custom attribute on the platform (idempotent)

UPDATE path (slug found):
    1. Apply only the fields you passed; leave the rest untouched
    2. Re-validate the platform if either token or account_id changed
    3. Skip bot_mode provisioning (assumed done at create time; re-run a
       create-side invocation if the platform account_id changes)

Secrets are encrypted at rest via `crypto.encrypt_optional`. Empty or unset
tokens land as SQL NULL, not as an encrypted empty string.
"""

from __future__ import annotations

import argparse
import asyncio
import secrets
from pathlib import Path
from typing import Optional

import httpx
from sqlalchemy import select

from app.config import settings
from app.core import crypto
from app.db.models.tenant import Tenant
from app.db.session import get_session
from app.integrations.mori_connect import MoriConnectClient


PLACEHOLDER_PROMPT = (
    "You are a helpful customer service assistant for {name}. "
    "Be concise, polite, and direct. If you cannot help or the customer "
    "asks for a human, escalate."
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Create or update a Receptionist tenant (upsert by slug).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--slug", required=True, help="URL-safe identifier (e.g. 'alabali')")

    # All other fields are optional. On CREATE, --name and --mori-connect-account-id
    # are required (checked in main() after the lookup so we can give a clearer
    # error than argparse would).
    p.add_argument("--name", default=None, help="Display name (required on CREATE)")
    p.add_argument(
        "--mori-connect-account-id",
        type=int,
        default=None,
        help="The tenant's account_id on the inbox platform (required on CREATE)",
    )
    p.add_argument(
        "--mori-connect-api-token",
        default=None,
        help="user_access_token from the tenant's profile settings on the inbox platform",
    )
    p.add_argument(
        "--prompt-file",
        type=Path,
        default=None,
        help="Path to a text file with the system prompt",
    )
    p.add_argument(
        "--mori-connect-base-url",
        default=None,
        help="Override inbox platform base URL (default: settings.MORI_CONNECT_BASE_URL)",
    )
    p.add_argument(
        "--notify-admin-on-message",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Per-message admin-notification mode",
    )
    p.add_argument(
        "--inbox-id",
        type=int,
        default=None,
        help=(
            "Optional. If given, attach the auto-created Agent Bot to this "
            "inbox in the tenant's Chatwoot account. Skip to assign manually "
            "in the Chatwoot UI later."
        ),
    )
    p.add_argument(
        "--bot-endpoint",
        default=None,
        help=(
            "Override the Agent Bot's outgoing_url. Defaults to "
            "RECEPTIONIST_PUBLIC_URL + /api/mori-connect?token=… "
            "from settings."
        ),
    )
    p.add_argument(
        "--medusa-api-url",
        default=None,
        help=(
            "The tenant's Medusa storefront base URL, e.g. "
            "'https://api.alabali.store'. Enables product sync into RAG."
        ),
    )
    p.add_argument(
        "--medusa-api-key",
        default=None,
        help=(
            "Medusa publishable key (pk_...) for /store API auth. "
            "Stored encrypted at rest."
        ),
    )
    return p.parse_args()


def _build_bot_endpoint(webhook_token: str, override: Optional[str]) -> str:
    """Construct the Agent Bot URL that Chatwoot will POST to.

    Override wins; otherwise build from settings + the tenant's own webhook
    token: <public-url>/api/mori-connect?token=<tenant_token>.

    The token is per-tenant (tenants.webhook_token) — NOT a shared secret —
    so the webhook handler authenticates AND identifies the tenant in one
    step. A leaked tenant token only exposes that tenant.
    """
    if override:
        return override
    base = (settings.RECEPTIONIST_PUBLIC_URL or "").rstrip("/")
    if not base:
        raise SystemExit(
            "❌ RECEPTIONIST_PUBLIC_URL must be set in .env to auto-register "
            "the Agent Bot, OR pass --bot-endpoint explicitly."
        )
    if not webhook_token:
        raise SystemExit(
            "❌ Tenant has no webhook_token. Re-run manage_tenant.py to "
            "generate one (CREATE path always does)."
        )
    return f"{base}/api/mori-connect?token={webhook_token}"


async def validate_mori_connect(client: MoriConnectClient, account_id: int) -> None:
    """Hit GET /accounts/{id} to confirm both token + account_id are valid."""
    url = f"{client.base_url}/api/v1/accounts/{account_id}"
    async with httpx.AsyncClient(timeout=10.0) as http:
        r = await http.get(url, headers=client._headers())
    if r.status_code != 200:
        raise SystemExit(
            f"❌ Inbox platform validation failed (HTTP {r.status_code}). "
            f"Check --mori-connect-account-id and --mori-connect-api-token. "
            f"Body: {r.text[:200]}"
        )


def load_prompt(prompt_file: Optional[Path], name: str) -> str:
    """Read a prompt file, or return the placeholder if not given."""
    if prompt_file is None:
        print("⚠ Using placeholder prompt. Edit later by re-running with --prompt-file.")
        return PLACEHOLDER_PROMPT.format(name=name)
    if not prompt_file.is_file():
        raise SystemExit(f"❌ Prompt file not found: {prompt_file}")
    text = prompt_file.read_text(encoding="utf-8").strip()
    if not text:
        raise SystemExit(f"❌ Prompt file is empty: {prompt_file}")
    return text


async def create_tenant(args: argparse.Namespace) -> None:
    """CREATE path — slug doesn't exist yet."""
    if not args.name:
        raise SystemExit("❌ --name is required when creating a new tenant.")
    if args.mori_connect_account_id is None:
        raise SystemExit("❌ --mori-connect-account-id is required when creating a new tenant.")

    prompt = load_prompt(args.prompt_file, args.name)

    # Validate Chatwoot only if we have a token to validate with.
    if args.mori_connect_api_token:
        client = MoriConnectClient(
            base_url=args.mori_connect_base_url,
            api_token=args.mori_connect_api_token,
        )
        await validate_mori_connect(client, args.mori_connect_account_id)
        print(f"✓ Inbox platform account {args.mori_connect_account_id} reachable")
    else:
        client = None
        print("⚠ No inbox platform token provided; skipping validation + bot_mode provisioning.")
        print("  Bot can't reply for this tenant until a token is added.")

    # Generate the per-tenant webhook auth token. This is BOTH the Chatwoot
    # bot URL's `?token=` value AND the lookup key our webhook handler uses
    # to identify the tenant. 43 URL-safe chars from secrets.token_urlsafe(32).
    webhook_token = secrets.token_urlsafe(32)

    with get_session() as db:
        tenant = Tenant(
            slug=args.slug,
            name=args.name,
            mori_connect_account_id=args.mori_connect_account_id,
            mori_connect_api_token_enc=crypto.encrypt_optional(args.mori_connect_api_token),
            prompt_template=prompt,
            notify_admin_on_message=(
                args.notify_admin_on_message if args.notify_admin_on_message is not None else True
            ),
            webhook_token=webhook_token,
            medusa_api_url=args.medusa_api_url,
            medusa_api_key_enc=crypto.encrypt_optional(args.medusa_api_key),
        )
        db.add(tenant)
        db.flush()
        tenant_id = tenant.id
    print(f"✓ Tenant created: id={tenant_id} slug={args.slug}")
    if args.medusa_api_url:
        print(f"✓ Medusa configured: {args.medusa_api_url}")

    if client is not None:
        result = await client.ensure_bot_mode_attribute(args.mori_connect_account_id)
        if result is None:
            print("✓ Custom attribute 'bot_mode' already exists on the inbox platform")
        else:
            print("✓ Custom attribute 'bot_mode' provisioned on the inbox platform")

        await _ensure_agent_bot(args, client, tenant_id)

    print(f"\n✅ Tenant '{args.slug}' created successfully.")


async def _ensure_agent_bot(
    args: argparse.Namespace, client: MoriConnectClient, tenant_id
) -> None:
    """Create the Agent Bot in Chatwoot if this tenant doesn't have one yet.
    Idempotent — safe to call on both CREATE and UPDATE paths.

    Backfills `webhook_token` if missing (tenant predates the per-tenant
    token rollout) so older rows can still get an Agent Bot URL."""
    with get_session() as db:
        row = db.execute(
            select(Tenant).where(Tenant.id == tenant_id)
        ).scalar_one()
        existing_bot_id = row.mori_connect_agent_bot_id
        tenant_name = row.name
        mori_connect_account_id = row.mori_connect_account_id
        webhook_token = row.webhook_token
        if not webhook_token:
            webhook_token = secrets.token_urlsafe(32)
            row.webhook_token = webhook_token
            print("✓ Backfilled missing webhook_token for this tenant.")

    if existing_bot_id is not None:
        print(f"✓ Agent Bot already registered (id={existing_bot_id}) — skipping create")
        await _attach_bot_to_inbox(
            args, client, mori_connect_account_id, existing_bot_id, tenant_name
        )
        return

    try:
        bot_endpoint = _build_bot_endpoint(webhook_token, args.bot_endpoint)
    except SystemExit as e:
        print(f"⚠ Skipping Agent Bot auto-registration: {e}")
        return

    bot = await client.create_agent_bot(
        account_id=mori_connect_account_id,
        name=f"{tenant_name} Receptionist",
        outgoing_url=bot_endpoint,
        description=f"AI receptionist for {tenant_name}. Managed by AlaBali.",
    )
    bot_id = bot.get("id")
    # Chatwoot returns the bot's own api_access_token in this response. It's
    # only shown here — there's no way to re-fetch it later. We MUST capture
    # it now so the runtime can post replies as the bot (sender.type =
    # 'agent_bot') rather than as the operator user (sender.type = 'user').
    bot_access_token = bot.get("access_token")
    print(f"✓ Agent Bot created on the inbox platform: id={bot_id}")
    if not bot_access_token:
        print(
            "⚠ Chatwoot did not return a bot access_token. The bot will fall "
            "back to the user token at runtime, which causes the self-echo "
            "bug. Re-run after upgrading or check the API response shape."
        )

    with get_session() as db:
        row = db.execute(
            select(Tenant).where(Tenant.id == tenant_id)
        ).scalar_one()
        row.mori_connect_agent_bot_id = bot_id
        if bot_access_token:
            row.mori_connect_bot_token_enc = crypto.encrypt_optional(bot_access_token)

    await _attach_bot_to_inbox(args, client, mori_connect_account_id, bot_id, tenant_name)


async def _attach_bot_to_inbox(
    args: argparse.Namespace,
    client: MoriConnectClient,
    account_id: int,
    bot_id: int,
    tenant_name: str,
) -> None:
    """Attach the bot to an inbox. Priority order:
       1. Explicit --inbox-id wins.
       2. Otherwise list inboxes; auto-attach if there's exactly one.
       3. If multiple inboxes, refuse to guess and print the list so the
          user can re-run with --inbox-id <N>.
    """
    if args.inbox_id is not None:
        await client.set_agent_bot_on_inbox(account_id, args.inbox_id, bot_id)
        print(f"✓ Agent Bot attached to inbox {args.inbox_id}")
        return

    try:
        inboxes = await client.list_inboxes(account_id)
    except Exception as e:
        print(f"⚠ Could not list inboxes ({e}). Attach manually in the inbox platform UI.")
        return

    if not inboxes:
        print(
            "ⓘ No inboxes found in this Chatwoot account yet. Create one in "
            "Chatwoot UI, then re-run with --inbox-id <N>."
        )
        return

    if len(inboxes) == 1:
        only = inboxes[0]
        await client.set_agent_bot_on_inbox(account_id, only["id"], bot_id)
        print(f"✓ Agent Bot attached to inbox {only['id']} ({only.get('name')})")
        return

    print(
        f"ⓘ Multiple inboxes found — won't guess. Re-run with --inbox-id <N>:"
    )
    for ib in inboxes:
        print(f"    {ib['id']:>6}  {ib.get('name', '<no name>')}  ({ib.get('channel_type', '?')})")


async def update_tenant(args: argparse.Namespace, tenant: Tenant) -> None:
    """UPDATE path — slug exists. Patch only the fields that were passed."""
    changed: list[str] = []

    if args.name is not None:
        tenant.name = args.name
        changed.append("name")

    if args.mori_connect_account_id is not None and args.mori_connect_account_id != tenant.mori_connect_account_id:
        tenant.mori_connect_account_id = args.mori_connect_account_id
        changed.append("mori_connect_account_id")

    if args.mori_connect_api_token is not None:
        tenant.mori_connect_api_token_enc = crypto.encrypt_optional(args.mori_connect_api_token)
        changed.append("mori_connect_api_token")

    if args.prompt_file is not None:
        tenant.prompt_template = load_prompt(args.prompt_file, tenant.name)
        changed.append("prompt_template")

    if args.notify_admin_on_message is not None:
        tenant.notify_admin_on_message = args.notify_admin_on_message
        changed.append("notify_admin_on_message")

    if args.medusa_api_url is not None:
        tenant.medusa_api_url = args.medusa_api_url
        changed.append("medusa_api_url")

    if args.medusa_api_key is not None:
        tenant.medusa_api_key_enc = crypto.encrypt_optional(args.medusa_api_key)
        changed.append("medusa_api_key")

    if not changed:
        print(f"✓ Tenant '{tenant.slug}' found, no fields to update.")
        # Fall through — we still run the best-effort agent-bot ensure step
        # below so re-running the script reconciles missing Chatwoot state.
    else:
        # Re-validate Chatwoot if creds touched. Use the (possibly just-updated)
        # token + account_id on the tenant for the check.
        if "mori_connect_api_token" in changed or "mori_connect_account_id" in changed:
            token = crypto.get_mori_connect_token(tenant) if tenant.mori_connect_api_token_enc else None
            if token:
                client = MoriConnectClient(base_url=args.mori_connect_base_url, api_token=token)
                await validate_mori_connect(client, tenant.mori_connect_account_id)
                print(f"✓ Inbox platform account {tenant.mori_connect_account_id} reachable")

        print(f"✓ Tenant '{tenant.slug}' updated: {', '.join(changed)}")

    # Best-effort: reconcile Chatwoot state. _ensure_agent_bot handles both
    # "no bot yet, create one" and "bot exists, just (re-)attach to inbox".
    # So re-running the script with --inbox-id N always works.
    if tenant.mori_connect_api_token_enc:
        token = crypto.get_mori_connect_token(tenant)
        if token:
            client = MoriConnectClient(base_url=args.mori_connect_base_url, api_token=token)
            await _ensure_agent_bot(args, client, tenant.id)


async def main() -> None:
    args = parse_args()

    with get_session() as db:
        existing = db.execute(
            select(Tenant).where(Tenant.slug == args.slug)
        ).scalar_one_or_none()

        if existing is None:
            print(f"⊕ Tenant '{args.slug}' not found — creating.")
            # `create_tenant` opens its own session. Fall through and call it
            # after this read-only lookup session closes.
        else:
            print(f"✎ Tenant '{args.slug}' found — updating.")
            await update_tenant(args, existing)
            return

    await create_tenant(args)


if __name__ == "__main__":
    asyncio.run(main())

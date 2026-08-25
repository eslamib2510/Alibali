# Onboarding a Tenant

How to hook a new client's messaging inbox into the receptionist.

Outcome after these steps: a customer messages the client's inbox, the receptionist replies, and admin sees the whole thread inside their inbox platform as normal.

---

## 1. Collect from the client

Three values, no more:

1. **Account ID**: the numeric ID for their workspace on the inbox platform. Usually visible in the platform URL after `/accounts/`.
2. **User Access Token**: from their profile page on the inbox platform (avatar, Profile Settings, Access Token). Used once, to create the bot on their behalf. Stored encrypted at rest.
3. **Inbox ID** (optional): the numeric ID of the inbox the bot should attach to. If they only have one inbox the script picks it automatically.

Also decide the receptionist prompt for this client. Copy `scripts/prompts/_TEMPLATE.txt` to `scripts/prompts/<slug>.txt` and tailor it.

## 2. Run the setup

One command from the repo root:

```bash
docker compose exec api python -m scripts.manage_tenant \
  --slug <slug> --name "<Display Name>" \
  --mori-connect-account-id <ACCOUNT_ID> \
  --mori-connect-api-token <USER_ACCESS_TOKEN> \
  --inbox-id <INBOX_ID> \
  --prompt-file scripts/prompts/<slug>.txt
```

Slug rules: URL-safe, lowercase, one word. Example: `alabali`.

## 3. What the script does

In one run:

1. Validates the user access token by hitting `GET /accounts/{id}` on the platform.
2. Inserts the tenants row with an encrypted token and a fresh per-tenant `webhook_token`.
3. Provisions the `bot_mode` custom attribute on the client's workspace (idempotent).
4. Creates a bot account on the client's workspace with `outgoing_url = <public URL>/api/mori-connect?token=<webhook_token>`.
5. Captures the bot's own access token from the create-bot response (only shown once) and stores it encrypted. The bot uses this token to post replies as itself, which prevents the self-loop where the bot reads its own message as a human takeover.
6. Attaches the bot to the inbox.

Nothing else needs to happen in the platform UI.

## 4. Verify it worked

```bash
docker compose exec api python -c "
from sqlalchemy import text
from app.db.session import engine
with engine.connect() as c:
    print(c.execute(text('''
      SELECT slug, mori_connect_account_id, mori_connect_agent_bot_id,
             mori_connect_api_token_enc IS NOT NULL AS has_api_token,
             mori_connect_bot_token_enc IS NOT NULL AS has_bot_token
      FROM tenants WHERE slug = :s
    '''), {'s': '<slug>'}).one())
"
```

All boolean columns should be `True` and `mori_connect_agent_bot_id` should be an integer.

Smoke test: send a message from the client's channel. Confirm the bot replies within a few seconds.

## 5. Updating a tenant later

Same script, upsert-by-slug. Pass only what you want to change:

```bash
# Rotate the user access token
docker compose exec api python -m scripts.manage_tenant \
  --slug <slug> --mori-connect-api-token <NEW_TOKEN>

# Swap the prompt
docker compose exec api python -m scripts.manage_tenant \
  --slug <slug> --prompt-file scripts/prompts/<slug>_v2.txt

# Attach the bot to a different inbox
docker compose exec api python -m scripts.manage_tenant \
  --slug <slug> --inbox-id <NEW_INBOX_ID>

# Switch to silent mode (no per-message admin notification)
docker compose exec api python -m scripts.manage_tenant \
  --slug <slug> --no-notify-admin-on-message
```


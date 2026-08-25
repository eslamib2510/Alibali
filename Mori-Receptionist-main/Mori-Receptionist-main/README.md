# Mori-Receptionist

Multi-tenant AI receptionist for businesses. Powers customer conversations across channels (WhatsApp, web widget, email, Instagram) via Mori-Connect, with memory, product knowledge (from Medusa), and tool-using agents.

Start simple. Built to scale.

---

## Setup

### 1. Configure environment

```bash
cp .env.example .env
```

Fill in these values in `.env`:

- `RECEPTIONIST_DATABASE_URL`, `RECEPTIONIST_DATABASE_DIRECT_URL`: Postgres connection strings
- `REDIS_URL`: Redis connection string
- `GEMINI_API_KEY`: LLM key
- `RECEPTIONIST_ENCRYPTION_KEY`: for encrypted tenant secrets at rest
- `RECEPTIONIST_PUBLIC_URL`: where this service lives on the internet

### 2. Build and run

```bash
docker compose up --build -d
```

Runs three services: `api` (FastAPI), `worker` (ARQ), `redis`. Re-run the same command after any code or `.env` change to rebuild and recreate the containers.

### 3. Onboard a tenant

See `docs/onboarding.md`.

---

## What it does

- Receives customer messages from Mori-Connect (via Agent Bot webhook)
- Loads per-tenant memory, business knowledge, recent conversation
- Calls an LLM with a defined toolset (FAQ search, product lookup, escalate, etc.)
- Posts the reply back through Mori-Connect
- Logs everything for observability and eval

One deployment serves many tenants. Each has its own prompt, tools config, and knowledge base.

---

## Stack

- **API + worker**: FastAPI + ARQ (Redis-backed background jobs)
- **DB**: Postgres + pgvector (for RAG)
- **Agent**: LangGraph (planned v2), currently direct Gemini call
- **LLM**: Gemini (Claude / OpenAI configurable per tenant later)
- **Chat channel**: Mori-Connect
- **Commerce**: Medusa (products fetched via API, embedded for RAG)

---

## Repo layout

```
Mori-Receptionist/
├── app/
│   ├── main.py               FastAPI entrypoint
│   ├── config.py             Pydantic settings
│   ├── api/                  HTTP routes
│   ├── core/                 Agent orchestration + crypto
│   ├── agent/                LangGraph state + graph (v2+)
│   ├── db/                   Models, session, repository, migrations
│   ├── integrations/         Mori-Connect, Gemini, Medusa clients
│   ├── tools/                Agent tools (search, escalate, ...)
│   ├── ingestion/            RAG write side (chunk, embed, store)
│   ├── retrieval/            RAG read side (hybrid search)
│   └── workers/              ARQ task definitions
├── docs/                     Architecture, tenancy, roadmap
├── scripts/                  Tenant management, seed
└── tests/                    Unit + integration
```

See `docs/roadmap.md` for the phased v1 through v5 plan.

---

## Migrations

```bash
alembic revision --autogenerate -m "<message>"
alembic upgrade head
```

Uses `RECEPTIONIST_DATABASE_DIRECT_URL` (direct, not pooled; pgbouncer transaction mode breaks DDL).

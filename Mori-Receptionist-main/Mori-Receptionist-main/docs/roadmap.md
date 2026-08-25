# Roadmap

Phased build. Each phase leaves the system deployable and useful.

## v1 — one tenant answering one question (current)

- FastAPI webhook receives Chatwoot event
- ARQ worker runs LLM call (Gemini)
- Posts reply back to Chatwoot
- Multi-tenant DB schema (tenants, customers, conversations, messages)
- Per-tenant webhook token auth
- Encrypted secrets at rest (Fernet)
- Voluntary escalation via ESCALATE: marker

**Status:** code ported from AlaBali backend. Needs deploy + first tenant test.

## v2 — Agentic RAG (knowledge base + LangGraph + first tool)

Merged v2+v4 into one phase so we skip the "just inject context" throwaway step.

- Add `knowledge` table with `pgvector` embedding column
- Admin endpoint to insert per-tenant FAQ text (chunk + embed)
- Hybrid search SQL (tsvector + vector + RRF)
- LangGraph state graph replaces the single-shot Gemini call
- First tool: `search_knowledge` (retrieves per-tenant chunks)
- Row-level security policies for tenant isolation

## v3 — Medusa product sync + live tools

- Background job: fetch products per tenant, embed title+description+variants
- Upsert into `knowledge` with `source_type='product'`
- Live tools: `get_product_stock`, `get_product_price` (real-time API, not embedded)
- Product webhooks trigger re-embed on update

## v4 — richer tools + memory

Escalation is already handled in `core/agent.py` via prompt-based `ESCALATE:` marker + human takeover detection. No `escalate_to_human` tool needed.

- `create_order` (Medusa checkout)
- `check_booking_availability` (calendar or Medusa slot table)
- `save_customer_fact` (long-term memory)
- Q&A generation from raw docs (Chatwoot Captain pattern)
- Reflexion-style self-critique on low-confidence replies

## v5 — multi-tenant hardening + admin UI

- Tenant admin dashboard (Next.js on separate repo)
- Per-tenant analytics (conversation count, escalation rate, avg latency)
- Model choice per tenant (Gemini / Claude / OpenAI)
- Rate limits per tenant
- Billing hooks

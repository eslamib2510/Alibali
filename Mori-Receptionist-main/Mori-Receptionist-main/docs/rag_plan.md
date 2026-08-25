# RAG + Agentic Plan

Detailed build plan for turning v1 (single-shot Gemini call) into v2 (RAG-grounded answers) and v3 (LangGraph agent with tools). Derived from Chatwoot Captain's architecture, LlamaIndex chunking patterns, pgvector 0.8+ best practices, and the small multi-tenant references (RedAgent, extrawest).

---

## Guiding decisions (locked)

| Decision | Choice | Reason |
|---|---|---|
| Vector index | pgvector HNSW | Better recall, no reindex maintenance, handles growth |
| Embedding model | `gemini-embedding-001` @ 1536 dims | Stays on Gemini stack, matches existing schema column, cheap ($0.15/1M) |
| Retrieval | Hybrid (vector + tsvector + RRF) | +8-15% accuracy over pure vector, one Postgres query |
| Chunking | Sentence-aware, 500 tokens / 50 overlap | LlamaIndex default, works for FAQ and product descriptions |
| Multi-tenancy | Single table, `tenant_id` column, Postgres RLS | Standard multi-tenant SaaS pattern |
| Agent framework | LangGraph (v3) | Stateful graph, tool nodes, human-in-loop ready |
| Ingest strategy | LLM-generated Q&A from raw docs (Chatwoot Captain pattern) | Better retrieval than raw chunks: query = question, doc = answer |

---

## Phase 2 — RAG grounding (no agent yet)

Goal: bot's replies are grounded in per-tenant knowledge. Still a single LLM call, no tool loop.

### 2.1 Schema

```sql
-- knowledge chunks (FAQ, product descriptions, admin-uploaded docs)
CREATE TABLE knowledge (
    id             UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id      UUID NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
    source_type    TEXT NOT NULL CHECK (source_type IN ('faq', 'product', 'document')),
    source_ref     TEXT,                       -- e.g. medusa product id, doc id
    title          TEXT,
    content        TEXT NOT NULL,
    content_tsv    tsvector GENERATED ALWAYS AS (to_tsvector('english', content)) STORED,
    embedding      vector(1536) NOT NULL,      -- gemini-embedding-001 truncated to 1536
    metadata       JSONB DEFAULT '{}',
    created_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at     TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- One HNSW index across all tenants; queries filter by tenant_id
CREATE INDEX ix_knowledge_embedding
    ON knowledge USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- Full-text search index for the lexical half of hybrid search
CREATE INDEX ix_knowledge_tsv
    ON knowledge USING gin (content_tsv);

-- Tenant scan (used by RLS + explicit WHERE)
CREATE INDEX ix_knowledge_tenant
    ON knowledge (tenant_id, source_type);

-- Enable row-level security
ALTER TABLE knowledge ENABLE ROW LEVEL SECURITY;

CREATE POLICY tenant_isolation ON knowledge
    USING (tenant_id = current_setting('app.tenant_id', true)::uuid);
```

At app startup or per-session: `SET LOCAL app.tenant_id = '<tenant_uuid>'`. RLS then guarantees isolation even if a WHERE is forgotten.

### 2.2 File additions

```
app/
├── db/
│   ├── models/
│   │   └── knowledge.py             ← Knowledge SQLAlchemy model
│   └── migrations/versions/
│       └── 2026_07_25_XXXX-knowledge_table.py
├── integrations/
│   └── gemini.py                    ← extend with `embed(text) -> list[float]`
├── services/
│   ├── chunker.py                   ← sentence-splitter (~100 lines, copy from LlamaIndex)
│   ├── faq_generator.py             ← LLM turns raw doc into Q&A pairs (Chatwoot Captain pattern)
│   ├── ingestor.py                  ← orchestrates: doc → chunks OR Q&A → embed → upsert
│   └── retriever.py                 ← hybrid search: vector + tsvector + RRF
├── api/
│   └── knowledge.py                 ← admin endpoint: POST /api/knowledge (add FAQ)
└── core/
    └── agent.py                     ← wire retriever call BEFORE LLM (context injection)
```

### 2.3 Embed function

```python
# app/integrations/gemini.py (addition)
from google.genai import types

def embed(text: str, dims: int = 1536) -> list[float]:
    """Return an embedding for `text`. 1536 dims = matches our schema column.
    Uses Matryoshka truncation, no quality loss vs 3072 default."""
    resp = _client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
        config=types.EmbedContentConfig(output_dimensionality=dims),
    )
    return resp.embeddings[0].values
```

Batch variant for ingestion (embed many at once, cheaper):
```python
def embed_batch(texts: list[str], dims: int = 1536) -> list[list[float]]:
    ...
```

### 2.4 Chunker (copy from LlamaIndex, ~100 lines)

- Sentence-aware split (respect sentence boundaries, don't cut mid-sentence)
- Target: 500 tokens per chunk, 50 token overlap
- Preserve small chunks; don't force-fill to 500 if the source is shorter
- Reference: `llama_index/core/node_parser/text/sentence.py`

### 2.5 FAQ generator (Chatwoot Captain's key insight)

Instead of embedding raw doc chunks, use an LLM to turn each chunk into 3-5 Q&A pairs, then embed the QUESTIONS. Retrieval is much better because customer queries look like questions, not statements.

```python
# app/ingestion/faq_generator.py
FAQ_PROMPT = """
Given the following business content, generate 3-5 concise question/answer pairs
a customer might ask, with answers grounded strictly in the source. Output JSON:
[{"q": "...", "a": "..."}, ...]

Source:
{content}
"""

def generate_faqs(content: str) -> list[dict]:
    ...  # single Gemini call, parse JSON
```

The Q gets embedded and used for retrieval matching. The A becomes the `content` field returned to the LLM as context.

### 2.6 Hybrid retrieval

One SQL query. Vector similarity + full-text search, merged via Reciprocal Rank Fusion (RRF). ~15 lines of SQL.

```sql
-- app/retrieval/retriever.py sends this
WITH vector_hits AS (
    SELECT id, RANK() OVER (ORDER BY embedding <=> $query_vec) AS rank
    FROM knowledge
    WHERE tenant_id = $tenant_id
    ORDER BY embedding <=> $query_vec
    LIMIT 20
),
fts_hits AS (
    SELECT id, RANK() OVER (ORDER BY ts_rank_cd(content_tsv, plainto_tsquery('english', $query_text)) DESC) AS rank
    FROM knowledge
    WHERE tenant_id = $tenant_id
      AND content_tsv @@ plainto_tsquery('english', $query_text)
    LIMIT 20
)
SELECT k.id, k.title, k.content,
       COALESCE(1.0 / (60 + v.rank), 0) + COALESCE(1.0 / (60 + f.rank), 0) AS rrf_score
FROM knowledge k
LEFT JOIN vector_hits v ON k.id = v.id
LEFT JOIN fts_hits    f ON k.id = f.id
WHERE (v.id IS NOT NULL OR f.id IS NOT NULL)
  AND k.tenant_id = $tenant_id
ORDER BY rrf_score DESC
LIMIT 5;
```

Over-fetch 20 per method, fuse with RRF constant `60` (industry default). Return top 5.

### 2.7 Enable iterative HNSW scan (pgvector 0.8+)

For small tenants where the top-N vector hits get filtered out by `tenant_id`:

```python
# In session.py, per connection
db.execute("SET hnsw.iterative_scan = 'relaxed_order'")
db.execute("SET hnsw.max_scan_tuples = 20000")
```

This lets HNSW keep walking the graph if the initial candidates all belong to other tenants.

### 2.8 Agent integration (still single-shot)

Modify `core/agent.py` right before the Gemini call:

```python
# BEFORE calling generate_reply(...)
context_chunks = retriever.search(
    tenant_id=tenant_id,
    query=event.content,
    limit=5,
)
context_block = "\n---\n".join(f"[{c.title}]\n{c.content}" for c in context_chunks)

full_system_prompt = f"""
{tenant_prompt}

Relevant knowledge (use this to ground your answer, cite implicitly by paraphrasing):
{context_block}

{_ESCALATION_INSTRUCTIONS}
""".strip()
```

Still one LLM call. Just better grounded. Zero LangGraph yet.

### 2.9 Ingest endpoints (admin-only)

```
POST /api/knowledge          - insert new FAQ (raw text → generate Q&As → embed → store)
POST /api/knowledge/product  - sync from Medusa (fetch → embed title+desc → store)  [v3 uses this]
DELETE /api/knowledge/:id
GET /api/knowledge?tenant_id=X
```

Auth via `deps.get_admin` (built when needed). Skipping tenant self-serve UI for now.

### 2.10 Testing v2

Manual acceptance criteria:
- Insert 20 FAQ entries for AlaBali tenant
- Send Chatwoot message: "do you have ice baths in stock?" → bot answer references FAQ content
- Send message about a topic NOT in FAQ → bot either says "not sure, let me connect you" or escalates
- Verify: second tenant's messages never retrieve first tenant's chunks (test with two tenants)

---

## Phase 3 — Medusa product sync

Goal: products from Medusa flow into `knowledge` table automatically.

### 3.1 Files

```
app/
├── integrations/
│   └── medusa.py                    ← client: fetch products, single product
└── services/
    └── product_sync.py              ← job: fetch all → embed → upsert into knowledge

scripts/
└── sync_products.py                 ← CLI wrapper: `python -m scripts.sync_products --tenant alabali`
```

### 3.2 Approach

- Per tenant, hit Medusa `/store/products?limit=1000` (paginate)
- For each product, build text: `f"{title}. {description}. Variants: {variants}. Tags: {tags}. Price: {price}"`
- Embed once, upsert into `knowledge` with `source_type='product'` and `source_ref=<medusa_product_id>`
- Idempotent: update existing rows instead of duplicating

### 3.3 Trigger options (choose one)

- **CLI on demand** — you run `sync_products.py` when catalog changes (v3.0)
- **Cron** — ARQ scheduled task nightly (v3.1)
- **Webhook** — Medusa `product.updated` → single-product re-embed (v3.2, optimal)

Start with CLI. Cron next. Webhook when scale demands it.

### 3.4 Live vs embedded

- **Embed**: title, description, tags, category (static-ish, safe to embed)
- **DON'T embed**: stock levels, price (change constantly)
- For live data, the agent will call a tool at v4 (`get_product_stock`, `get_product_price`)

---

## Phase 4 — Agentic (LangGraph with tools)

Goal: replace the single-shot Gemini call with a proper agent that can decide "search knowledge OR call a live tool OR escalate."

### 4.1 State schema

```python
# app/agent/state.py
from typing import TypedDict, Annotated
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    tenant_id: str
    conversation_id: str
    messages: Annotated[list, add_messages]      # user + assistant + tool
    retrieved_context: list[dict]                # populated by retrieve node
    escalation_reason: str | None                # populated by escalate node
```

### 4.2 Graph shape

```
     [entry]
        ↓
  ┌──> [agent_step]  ← LLM decides: reply, call tool, or escalate
  │        │
  │  ┌─────┴──────┬──────────────┐
  │  ↓            ↓              ↓
  │ [tool_node] [escalate]   [respond]
  │  │            │              │
  └──┘         [END]           [END]
```

- `agent_step` = call LLM with current state + tool descriptors, model chooses action
- `tool_node` = executes tool, appends result to messages, loops back to agent_step
- `escalate` = LLM said escalate → set state, exit graph
- `respond` = LLM produced a customer-facing reply → exit graph

### 4.3 Tool set (start small, add over time)

| Tool | Signature | Notes |
|---|---|---|
| `search_knowledge` | `(query: str) -> list[Chunk]` | Wraps `retriever.search`. Auto-scoped to tenant via state. |
| `get_product_stock` | `(product_id: str) -> int` | Live Medusa call, not embedded |
| `get_product_price` | `(product_id: str) -> Price` | Same |
| `check_booking_slot` | `(date: str) -> list[str]` | Optional per-tenant |
| `save_customer_fact` | `(fact: str) -> None` | Long-term memory write |

Tools defined as LangGraph `ToolNode`. Each is one file in `app/tools/`.

### 4.4 Escalation stays in graph, not tool

Escalation is a graph node, not a tool. The LLM's SYSTEM prompt tells it "if you need a human, output `ESCALATE: <reason>`". We parse the output; if it matches, we route to the `escalate` node instead of `respond`. Keeps escalation behavior identical to v1/v2.

### 4.5 Files

```
app/agent/
├── __init__.py
├── state.py            ← AgentState TypedDict
├── graph.py            ← build_graph() function
├── nodes.py            ← agent_step, respond, escalate node funcs
└── prompts.py          ← system prompt template with tool descriptors

app/tools/
├── __init__.py
├── base.py             ← Tool base class + registry
├── search_knowledge.py
├── get_product_stock.py
├── get_product_price.py
└── save_customer_fact.py
```

### 4.6 Wiring

Replace the current Gemini call in `core/agent.py` Phase 2 with:

```python
graph = build_graph(tenant_id=tenant_id, tools=load_tools_for_tenant(tenant_id))
final_state = await graph.ainvoke({
    "tenant_id": tenant_id,
    "conversation_id": str(conversation_pk),
    "messages": history_as_langchain_messages(history) + [HumanMessage(event.content)],
    "retrieved_context": [],
    "escalation_reason": None,
})
```

Read `final_state` for `escalation_reason` vs the assistant's last message → same downstream flow (send reply, post note, toggle status).

### 4.7 Per-tenant tool config (Chatwoot Captain-inspired)

Store tools as DB rows so tenants can enable/disable without a deploy:

```sql
CREATE TABLE tenant_tools (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    tenant_id UUID REFERENCES tenants(id),
    tool_name TEXT NOT NULL,                -- 'get_product_stock', 'check_booking_slot', ...
    enabled BOOLEAN DEFAULT true,
    config JSONB DEFAULT '{}',              -- tool-specific params
    UNIQUE (tenant_id, tool_name)
);
```

`load_tools_for_tenant(tenant_id)` reads this and returns only enabled tools.

---

## Phase 5 — Hardening + admin UX

- **RLS enforcement middleware** — set `app.tenant_id` per session automatically
- **Admin UI** (Next.js separate repo) — FAQ CRUD, tenant analytics, tool toggles
- **Evals** — a small test suite of (query, expected_topic) pairs per tenant, run nightly
- **Reranking** (optional, if accuracy hurts) — cross-encoder rerank on top-20 from retriever, pick top-5. Use `bge-reranker-base` self-hosted or Cohere Rerank API.
- **Conversation memory** — beyond last 10 messages, embed past resolved conversations for long-term memory retrieval

---

## What to copy from where

| Piece | Source | Notes |
|---|---|---|
| Sentence chunker | LlamaIndex `SentenceSplitter` | Vendor ~100 lines, don't `pip install llama-index` |
| Hybrid search SQL | dev.to hybrid search tutorial | Paste + adapt tenant_id filter |
| FAQ generation from docs | Chatwoot Captain `FaqGeneratorService` | Steal the pattern, write in Python |
| Multi-tenant skeleton | RedAgent | Folder structure + service layer style |
| LangGraph state + tool node | LangGraph `customer-support-bot` example | Copy the graph shape |
| RLS policy | Tigerdata multi-tenant RAG guide | One-off, well-documented |
| Custom tools as DB rows | Chatwoot Captain `Captain::CustomTool` | Table schema idea |

**Zero framework forking. All small, focused copies into `app/ingestion/`, `app/retrieval/`, `app/tools/`, etc.**

---

## Build order

Ordered for shippability — each item leaves the system deployable and useful:

1. `knowledge` table migration + `Knowledge` model (v2.1)
2. `gemini.embed()` function (v2.3)
3. `chunker.py` (v2.4)
4. `retriever.py` hybrid search (v2.6)
5. `ingestor.py` + `POST /api/knowledge` admin endpoint (v2.9)
6. Wire retriever into `core/agent.py` as context injection (v2.8)
7. Test with real AlaBali FAQ (v2.10)
8. `faq_generator.py` LLM Q&A pattern (v2.5) — swap out raw chunking
9. Medusa integration + `product_sync.py` (v3.1-3.3)
10. LangGraph state + graph skeleton (v4.1-4.2)
11. `search_knowledge` tool + wire into graph (v4.3)
12. Replace agent.py call with graph.ainvoke (v4.6)
13. Add tools one at a time (v4.3 remaining)
14. RLS, admin UI, evals (v5)

Steps 1-7 = v2 shipped. Adds real value with zero framework debt. Steps 8-14 = v3+ evolution.

---

## Cost sanity check

- Embedding: 1000 FAQ entries per tenant × ~200 tokens each × 100 tenants = 20M tokens × $0.15/M = **$3 to embed everything once**
- Per query embedding: ~50 tokens × 10k queries/day = 500k tokens/day × $0.15/M = **$0.075/day**
- LLM (Gemini 3.1 flash-lite): ~$0.10/1M input, ~$0.40/1M output. RAG context adds ~1k tokens per turn. Negligible.

Total: pennies per tenant per day. Scale won't hurt.

---

## Sources

- Chatwoot Captain architecture — https://deepwiki.com/chatwoot/chatwoot/9.1-captain-ai-system
- Gemini Embedding — https://developers.googleblog.com/gemini-embedding-available-gemini-api/
- pgvector 0.8 iterative scan — https://www.thenile.dev/blog/pgvector-080
- Hybrid search RRF (SQL) — https://dev.to/lpossamai/building-hybrid-search-for-rag-combining-pgvector-and-full-text-search-with-reciprocal-rank-fusion-6nk
- pgvector limitations + filtering — https://www.paradedb.com/learn/postgresql/pgvector-limitations
- LangGraph customer support — https://www.lancedb.com/blog/agentic-rag-using-langgraph-building-a-simple-customer-support-autonomous-agent
- Multi-tenant on Postgres — https://www.tigerdata.com/blog/building-multi-tenant-rag-applications-with-postgresql-choosing-the-right-approach
- RedAgent (shape reference) — https://github.com/pratiksontakke/redagent-backend
- LlamaIndex chunking source — https://github.com/run-llama/llama_index/tree/main/llama-index-core/llama_index/core/node_parser/text

"""AgentState — the single dict the LangGraph passes between nodes.

TypedDict rather than a plain dict so any node that misspells a key gets caught
by static analysis instead of writing to a phantom field the graph then never
reads.

`messages` is annotated with `add_messages` so returning `{"messages": [x]}`
from a node appends x to the existing list instead of replacing the whole
history — the standard LangGraph reducer pattern for chat state.

`tenant_id` is set once at graph entry and never overwritten. Tools that query
per-tenant data (search_knowledge, get_product_stock) close over it from
here rather than accepting it as an LLM-supplied argument — if the model got
to name the tenant, that IS a cross-tenant leak by design.
"""

from __future__ import annotations

from typing import Annotated, TypedDict

from langgraph.graph.message import add_messages


class AgentState(TypedDict):
    tenant_id: str  # never mutated after graph entry — see module docstring
    messages: Annotated[list, add_messages]
    escalation_reason: str | None

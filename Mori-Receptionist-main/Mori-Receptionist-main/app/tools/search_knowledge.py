"""search_knowledge — the RAG lookup tool the agent can call mid-conversation.

Wraps `app.retrieval.retriever.search()` in a LangChain tool. The retriever is
sync + blocking (query embedding + Postgres hit); the graph is async, so we
dispatch through `asyncio.to_thread` rather than calling it directly and
freezing the event loop for every concurrent chat.

Tenant isolation
----------------
`tenant_id` is bound at construction time via a closure, NOT passed as a tool
argument the LLM can supply. If the model chose the tenant_id, prompt
injection could redirect a search into another tenant's knowledge base. The
`build_search_knowledge()` factory returns a per-request tool whose tenant is
already fixed.

Return shape
------------
Returns the retrieved chunks as a formatted string (title + content, one per
block, "no results" if empty) rather than a list of dicts. LangChain's tool
protocol wants a string back — the model reads this as context in the next
turn.
"""

from __future__ import annotations

import asyncio
import logging

from langchain_core.tools import BaseTool, tool

from app.retrieval.retriever import format_context, search

logger = logging.getLogger(__name__)

# How many chunks each search call returns. Small enough to keep the model's
# context lean, big enough to cover multi-source questions ("what's the price
# AND when does it ship").
DEFAULT_LIMIT = 5


def build_search_knowledge(tenant_id: str) -> BaseTool:
    """Return a search_knowledge tool bound to one tenant.

    Called once per conversation turn from build_graph. The returned tool is
    what the LLM sees — its docstring becomes the tool description the model
    reads when deciding whether to call it, so keep it aimed at the model
    (what to search for, when to skip searching) rather than at the developer.
    """

    @tool
    async def search_knowledge(query: str) -> str:
        """Search this business's knowledge base for relevant information.

        Use this when the customer asks about:
        - Products, pricing, availability, specifications
        - Company policies (returns, shipping, hours, locations)
        - Anything specific to this business you don't already know

        Skip it for:
        - Greetings, small talk, thanks
        - General knowledge the model already has
        - Follow-up clarifications on something already searched this turn

        Args:
            query: Natural-language question the customer is trying to answer.
                Rewrite ambiguous or elliptical follow-ups into standalone
                questions before searching (e.g. "how about the blue one?"
                after "do you sell ice baths?" -> "do you sell blue ice baths?").
        """
        # Dispatch off the loop — same rule as everything in gemini.py.
        chunks = await asyncio.to_thread(
            search,
            tenant_id=tenant_id,
            query=query,
            limit=DEFAULT_LIMIT,
        )
        if not chunks:
            return "No matching knowledge found. Answer from the system prompt or escalate."
        return format_context(chunks)

    return search_knowledge

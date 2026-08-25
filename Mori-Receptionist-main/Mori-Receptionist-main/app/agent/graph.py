"""build_graph — assemble the LangGraph state machine for one conversation.

Shape:

    START -> agent_step -> (tool_needed?)-> tool_node -> agent_step -> ...
                              (no)      -> END

`agent_step` calls Gemini with the current messages + tool descriptors and
appends the model's reply (which may be text, a tool call, or the ESCALATE
marker) to state. The conditional edge routes:
- tool_calls present -> tool_node executes them, appends results, loops back
- ESCALATE: marker in the reply text -> writes state.escalation_reason and ends
- plain text -> ends, caller reads the last assistant message as the reply

Escalation stays as a string marker (not a tool call) so the existing
prompt-based flow from core/agent.py carries over unchanged — no need to
rewrite the tenant prompt or teach the model a new escalate() tool.
"""

from __future__ import annotations

import logging
from typing import Literal

from langchain_core.messages import AIMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from app.agent.state import AgentState
from app.config import settings
from app.tools.search_knowledge import build_search_knowledge

logger = logging.getLogger(__name__)


# Same escalation contract used by the pre-LangGraph agent in core/agent.py.
# The tenant's prompt gets this appended before Gemini sees it, so the LLM
# knows the exact single-line marker format that means "hand off to a human".
ESCALATION_INSTRUCTIONS = """
---
Escalation rules (apply to every turn):

Escalate to a human teammate when ANY of these are true:
  - The customer explicitly asks to speak to a person/agent/human.
  - The customer is clearly frustrated, angry, or distressed.
  - The request involves refunds, complaints, account changes, billing
    disputes, legal questions, medical advice, or anything you should not
    answer on the company's behalf.
  - You don't know the answer with reasonable confidence and the question
    really matters to the customer.

When escalating, reply with EXACTLY this format on a single line, nothing
else, no quotes, no preamble:

ESCALATE: <short reason in one short sentence>

Otherwise, reply normally as the receptionist. Use the search_knowledge tool
before answering anything you don't already know about the business.
"""


def parse_escalation(text: str) -> str | None:
    """If the model replied with an ESCALATE marker, return the reason. Else None."""
    first_line = (text or "").strip().splitlines()[0] if text else ""
    if first_line.upper().startswith("ESCALATE:"):
        return first_line.split(":", 1)[1].strip() or "(no reason given)"
    return None


def build_graph(*, tenant_id: str, tenant_prompt: str):
    """Return a compiled LangGraph for one conversation turn.

    Called per Chatwoot webhook, cheap enough that graph re-instantiation
    isn't worth caching — the LLM object holds no per-conversation state and
    the tool binds to this tenant only, so caching across tenants would leak.
    """
    search_tool = build_search_knowledge(tenant_id)
    tools = [search_tool]

    llm = ChatGoogleGenerativeAI(
        model=settings.RECEPTIONIST_DEFAULT_MODEL,
        google_api_key=settings.GEMINI_API_KEY,
        # Same conservative defaults as the pre-LangGraph gemini.generate_reply.
        temperature=0.6,
        max_output_tokens=600,
    ).bind_tools(tools)

    system_message = SystemMessage(
        content=tenant_prompt.rstrip() + ESCALATION_INSTRUCTIONS
    )

    async def agent_step(state: AgentState) -> dict:
        """Call the LLM with the current transcript + tools; return its reply.

        Prepends the tenant system prompt on every call rather than storing it
        in state so it can't accidentally be edited by later nodes or replayed
        across tenants when the graph is reused (it isn't, but defense in
        depth is free here).
        """
        response = await llm.ainvoke([system_message, *state["messages"]])
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> Literal["tool_node", "check_escalation"]:
        """Tool call -> execute tools. Otherwise check for escalation marker."""
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tool_node"
        return "check_escalation"

    def check_escalation(state: AgentState) -> dict:
        """Set escalation_reason if the reply starts with ESCALATE:.

        A separate node so we can trace it in LangGraph's UI later, and so
        the "check + end" logic doesn't hide inside a routing function that
        can't touch state.
        """
        last = state["messages"][-1]
        text = last.content if isinstance(last.content, str) else str(last.content)
        reason = parse_escalation(text)
        return {"escalation_reason": reason} if reason else {}

    graph = StateGraph(AgentState)
    graph.add_node("agent_step", agent_step)
    graph.add_node("tool_node", ToolNode(tools))
    graph.add_node("check_escalation", check_escalation)

    graph.add_edge(START, "agent_step")
    graph.add_conditional_edges(
        "agent_step",
        route_after_agent,
        {"tool_node": "tool_node", "check_escalation": "check_escalation"},
    )
    graph.add_edge("tool_node", "agent_step")
    graph.add_edge("check_escalation", END)

    return graph.compile()

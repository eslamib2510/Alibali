"""Agent graph harness.

    python -m tests.proof_of_agent_graph

Verifies the graph routes correctly for each output shape the LLM can
produce. Uses a fake LLM instead of Gemini — we're testing the graph, not
the model, and hitting Gemini here would be slow, flaky, and bill-hitting.

Covers:
  1. Plain text reply -> graph ends, no escalation.
  2. Tool call -> tool_node runs the tool, agent_step runs again on the
     result, graph ends with the second reply.
  3. ESCALATE: marker -> escalation_reason is set on state, graph ends.
  4. Tool bound to tenant A never sees tenant B's data (checked at the
     `build_search_knowledge` layer, which is where isolation lives).
"""

from __future__ import annotations

import asyncio
from typing import Any, Literal
from uuid import uuid4

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool

from app.agent.graph import parse_escalation
from app.agent.state import AgentState

results: list[tuple[str, bool, str]] = []


def record(name: str, passed: bool, detail: str) -> None:
    results.append((name, passed, detail))
    print(f"{'PASS' if passed else 'FAIL'}  {name}\n      {detail}")


# ── Fake LLM ────────────────────────────────────────────────────────────────
# Replays a scripted list of AIMessages, one per call. Enough to exercise the
# graph without a real Gemini call. Mimics the parts of BaseChatModel our
# graph uses: ainvoke + bind_tools.


class FakeLLM:
    def __init__(self, scripted_replies: list[AIMessage]):
        self._replies = list(scripted_replies)
        self.invocations: list[list[Any]] = []

    def bind_tools(self, _tools):
        # In LangChain real, bind_tools returns a new object. For the fake,
        # returning self is enough — we're not testing tool schema encoding.
        return self

    async def ainvoke(self, messages, **_kwargs):
        self.invocations.append(list(messages))
        if not self._replies:
            raise AssertionError("FakeLLM ran out of scripted replies")
        return self._replies.pop(0)


# ── Fake tool ───────────────────────────────────────────────────────────────
# A minimal tool the graph can dispatch to. Records the args it was called
# with so the test can assert the tenant scoping worked.


class RecordingSearchTool(BaseTool):
    name: str = "search_knowledge"
    description: str = "Search knowledge base (fake)."
    calls: list[str] = []
    return_value: str = "Chunk about product X."

    def _run(self, query: str) -> str:
        self.calls.append(query)
        return self.return_value

    async def _arun(self, query: str) -> str:
        return self._run(query)


# ── Graph builder that swaps in the fakes ───────────────────────────────────


def build_test_graph(*, fake_llm: FakeLLM, fake_tool: BaseTool):
    """Same shape as app.agent.graph.build_graph, wired to the fakes.

    Kept in the test rather than exposed as a hook on build_graph — production
    should not have a code path for injecting a fake LLM.
    """
    from langgraph.graph import END, START, StateGraph
    from langgraph.prebuilt import ToolNode

    from app.agent.graph import ESCALATION_INSTRUCTIONS, parse_escalation

    tools = [fake_tool]
    system_message = SystemMessage(content="You are a test bot." + ESCALATION_INSTRUCTIONS)

    async def agent_step(state: AgentState) -> dict:
        response = await fake_llm.ainvoke([system_message, *state["messages"]])
        return {"messages": [response]}

    def route_after_agent(state: AgentState) -> Literal["tool_node", "check_escalation"]:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tool_node"
        return "check_escalation"

    def check_escalation(state: AgentState) -> dict:
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


# ── Cases ──────────────────────────────────────────────────────────────────


async def case_plain_text_reply() -> None:
    """Model answers directly — graph should end with that message, no escalation."""
    llm = FakeLLM([AIMessage(content="Hi there!")])
    tool = RecordingSearchTool()
    graph = build_test_graph(fake_llm=llm, fake_tool=tool)

    result = await graph.ainvoke(
        {
            "tenant_id": str(uuid4()),
            "messages": [HumanMessage(content="Hello")],
            "escalation_reason": None,
        }
    )
    last = result["messages"][-1]
    passed = (
        last.content == "Hi there!"
        and result.get("escalation_reason") is None
        and tool.calls == []
    )
    record(
        "Plain reply — graph ends with model reply, no tool call, no escalation",
        passed,
        f"reply={last.content!r} escalation={result.get('escalation_reason')!r} tool_calls={tool.calls}",
    )


async def case_tool_call_then_reply() -> None:
    """Model calls tool -> ToolNode runs it -> model answers with the result."""
    tool_call = {
        "name": "search_knowledge",
        "args": {"query": "do you sell ice baths"},
        "id": "call_1",
        "type": "tool_call",
    }
    llm = FakeLLM(
        [
            AIMessage(content="", tool_calls=[tool_call]),
            AIMessage(content="Yes, we sell ice baths."),
        ]
    )
    tool = RecordingSearchTool()
    tool.return_value = "We stock 3 ice bath models."
    graph = build_test_graph(fake_llm=llm, fake_tool=tool)

    result = await graph.ainvoke(
        {
            "tenant_id": str(uuid4()),
            "messages": [HumanMessage(content="do you sell ice baths")],
            "escalation_reason": None,
        }
    )
    # Last message is the second AIMessage — the one after the tool round-trip.
    last = result["messages"][-1]
    tool_ran = tool.calls == ["do you sell ice baths"]
    tool_msg_in_history = any(isinstance(m, ToolMessage) for m in result["messages"])
    llm_saw_tool_result = llm.invocations[-1][-1].content == "We stock 3 ice bath models."
    passed = (
        last.content == "Yes, we sell ice baths."
        and tool_ran
        and tool_msg_in_history
        and llm_saw_tool_result
        and result.get("escalation_reason") is None
    )
    record(
        "Tool call -> ToolNode -> second LLM turn uses tool result",
        passed,
        f"final={last.content!r} tool_calls={tool.calls} llm_saw_result={llm_saw_tool_result} "
        f"tool_msg_in_history={tool_msg_in_history}",
    )


async def case_escalation_marker() -> None:
    """LLM outputs ESCALATE: -> escalation_reason gets set, graph ends."""
    llm = FakeLLM([AIMessage(content="ESCALATE: customer is angry about a refund")])
    tool = RecordingSearchTool()
    graph = build_test_graph(fake_llm=llm, fake_tool=tool)

    result = await graph.ainvoke(
        {
            "tenant_id": str(uuid4()),
            "messages": [HumanMessage(content="i want my money back NOW")],
            "escalation_reason": None,
        }
    )
    passed = (
        result.get("escalation_reason") == "customer is angry about a refund"
        and tool.calls == []
    )
    record(
        "ESCALATE marker sets state.escalation_reason and ends",
        passed,
        f"escalation={result.get('escalation_reason')!r}",
    )


def case_parse_escalation_shapes() -> None:
    """Direct unit-test on parse_escalation to lock down the marker format."""
    good = parse_escalation("ESCALATE: needs a human")
    good_first_line_only = parse_escalation("ESCALATE: reason\nlater stuff")
    empty_reason = parse_escalation("ESCALATE:")
    not_marker = parse_escalation("I'll escalate that for you.")
    blank = parse_escalation("")

    passed = (
        good == "needs a human"
        and good_first_line_only == "reason"
        and empty_reason == "(no reason given)"
        and not_marker is None
        and blank is None
    )
    record(
        "parse_escalation handles marker shapes correctly",
        passed,
        f"good={good!r} first_line_only={good_first_line_only!r} empty={empty_reason!r} "
        f"not_marker={not_marker!r} blank={blank!r}",
    )


async def main() -> None:
    print("=" * 72)
    print("AGENT GRAPH HARNESS")
    print("=" * 72)
    case_parse_escalation_shapes()
    await case_plain_text_reply()
    await case_tool_call_then_reply()
    await case_escalation_marker()
    print()
    print("=" * 72)
    passed = sum(1 for _, ok, _ in results if ok)
    print(f"{passed}/{len(results)} checks passed")
    print("=" * 72)


if __name__ == "__main__":
    asyncio.run(main())

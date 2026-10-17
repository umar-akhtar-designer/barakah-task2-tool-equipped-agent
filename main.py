"""Task 2: Equipping Agents with External Tools.

A LangChain tool-calling agent bound to two real tools:
  * calculator  — safe arithmetic evaluation (AST-based, no eval)
  * wikipedia    — live search against the Wikipedia REST API

Without an OPENAI_API_KEY the agent runs in a scripted offline mode that
still executes the real tools, so the whole loop is demonstrable offline.
"""

from __future__ import annotations

import ast
import os
import operator
import sys
from typing import Any, Dict

import requests
from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()

WIKIPEDIA_API = "https://en.wikipedia.org/api/rest_v1/page/summary/{title}"


@tool
def calculator(expression: str) -> str:
    """Evaluate an arithmetic expression, e.g. '(18 * 24) + 7'. Supports + - * / // % ** and parentheses."""
    allowed = {
        ast.Add: operator.add,
        ast.Sub: operator.sub,
        ast.Mult: operator.mul,
        ast.Div: operator.truediv,
        ast.FloorDiv: operator.floordiv,
        ast.Mod: operator.mod,
        ast.Pow: operator.pow,
        ast.USub: operator.neg,
        ast.UAdd: operator.pos,
    }

    def _eval(node: ast.AST) -> float:
        if isinstance(node, ast.Expression):
            return _eval(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
            return node.value
        if isinstance(node, ast.BinOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.left), _eval(node.right))
        if isinstance(node, ast.UnaryOp) and type(node.op) in allowed:
            return allowed[type(node.op)](_eval(node.operand))
        raise ValueError(f"Unsupported expression element: {ast.dump(node)}")

    try:
        result = _eval(ast.parse(expression, mode="eval"))
        return str(result)
    except Exception as exc:  # noqa: BLE001
        return f"Calculator error: {exc}"


@tool
def wikipedia_search(title: str) -> str:
    """Fetch the summary of a Wikipedia article by its title, e.g. 'Artificial_intelligence'."""
    try:
        resp = requests.get(
            WIKIPEDIA_API.format(title=title.replace(" ", "_")),
            headers={"User-Agent": "BarakahInternshipTask2/1.0"},
            timeout=10,
        )
        resp.raise_for_status()
        data = resp.json()
        extract = data.get("extract", "No summary found.")
        return f"{data.get('title', title)}: {extract}"
    except Exception as exc:  # noqa: BLE001
        return f"Wikipedia lookup failed for '{title}': {exc}"


TOOLS = [calculator, wikipedia_search]


def build_agent():
    """Return (agent_executor, prompt). Real OpenAI agent, or offline scripted agent."""
    from langchain_core.prompts import ChatPromptTemplate

    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a precise research assistant. Use tools when needed. "
                "Always cite the tool output in your final answer.",
            ),
            ("human", "{input}"),
            ("placeholder", "{agent_scratchpad}"),
        ]
    )

    api_key = os.getenv("OPENAI_API_KEY")
    if api_key:
        from langchain.agents import AgentExecutor, create_tool_calling_agent
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"), temperature=0)
        agent = create_tool_calling_agent(llm, TOOLS, prompt)
        return AgentExecutor(agent=agent, tools=TOOLS, verbose=True, max_iterations=5), prompt

    # Offline: scripted tool calls through a fake message list model.
    from langchain.agents import AgentExecutor, create_tool_calling_agent
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    class ScriptedToolModel(FakeMessagesListChatModel):
        """Fake model that satisfies the tool-binding contract by returning itself."""

        def bind_tools(self, tools, **kwargs):  # noqa: ANN001, ANN003
            return self

    print(
        "[offline mode] OPENAI_API_KEY not set — running a scripted agent that "
        "still executes the real calculator/Wikipedia tools.",
        file=sys.stderr,
    )
    scripted = ScriptedToolModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "calculator", "args": {"expression": "(18 * 24) + 7"}, "id": "call_1"}
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {"name": "wikipedia_search", "args": {"title": "Artificial intelligence"}, "id": "call_2"}
                ],
            ),
            AIMessage(content="Scripted final answer: (18 * 24) + 7 = 439, per the calculator tool."),
        ]
    )
    agent = create_tool_calling_agent(scripted, TOOLS, prompt)
    return AgentExecutor(agent=agent, tools=TOOLS, verbose=True, max_iterations=5), prompt


def run(query: str) -> Dict[str, Any]:
    executor, _ = build_agent()
    return executor.invoke({"input": query})


def main() -> None:
    query = (
        os.environ.get("AGENT_QUERY")
        or "What is (18 * 24) + 7? Then look up the Wikipedia summary of Artificial intelligence."
    )
    print(f"Query: {query}\n")
    result = run(query)
    print(f"\nFinal answer:\n{result.get('output', result)}")


if __name__ == "__main__":
    main()

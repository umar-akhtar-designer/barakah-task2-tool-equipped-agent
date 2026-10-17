"""Tests for Task 2 — offline, no API keys required."""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
os.environ.pop("OPENAI_API_KEY", None)

import main  # noqa: E402


def test_calculator_safe_math():
    assert main.calculator.invoke({"expression": "(18 * 24) + 7"}) == "439"
    assert main.calculator.invoke({"expression": "2 ** 10"}) == "1024"


def test_calculator_rejects_unsafe_input():
    result = main.calculator.invoke({"expression": "__import__('os').getcwd()"})
    assert result.startswith("Calculator error")


def test_wikipedia_tool_schema():
    schema = main.wikipedia_search.args_schema.model_json_schema()
    assert "title" in schema["properties"]


def test_agent_loop_executes_real_tools():
    result = main.run("What is (18 * 24) + 7?")
    assert "439" in result["output"]

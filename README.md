# Task 2 — Equipping Agents with External Tools

**Barakah TechLabs Internship · AI & Workflow Automation · AI Agent Developer**

A LangChain tool-calling agent bound to two typed `@tool` functions:

| Tool | What it does |
|---|---|
| `calculator` | Safely evaluates arithmetic via Python AST — no `eval()` |
| `wikipedia_search` | Live summary lookup against the Wikipedia REST API |

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Offline demo (no key) — scripted agent still runs the REAL tools
python main.py

# Live OpenAI tool-calling agent
cp .env.example .env   # add OPENAI_API_KEY
python main.py
```

## How tools turn an LLM into actionable software

1. **Typed function schema** — the `@tool` decorator derives a JSON schema
   (name, description, parameters) from the Python signature + docstring.
2. **Binding** — `llm.bind_tools([...])` injects those schemas into the model's
   API request so it can emit structured `tool_calls`.
3. **Execution loop** — `AgentExecutor` runs: model → tool call → tool result
   (as a `ToolMessage`) → model again, until the model answers without tools.

The agent never invents data: it must call `calculator` for math and
`wikipedia_search` for facts, and its final answer cites tool output.

## Offline mode

`ScriptedToolModel` (a `FakeMessagesListChatModel` subclass) emits predetermined
tool calls, and `AgentExecutor` executes the **real** tools — you can watch the
full agent loop work with zero API keys or cost.

## Tests

```bash
pytest -q
```

## Submission notes

- **Repository:** this repo
- **LinkedIn post:** see [`SUBMISSION_LINKEDIN.md`](SUBMISSION_LINKEDIN.md)

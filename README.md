# Recon

Multi-agent RAG system that turns competitor news into a weekly sales brief.

## How it works

Agents are wired together as a [LangGraph](https://langchain-ai.github.io/langgraph/) graph that passes a shared `AgentState` between nodes:

1. Fetch recent news for each tracked competitor (RSS via `feedparser`, tools exposed over `fastmcp`).
2. Index and retrieve relevant material from a Chroma vector store.
3. Draft the weekly sales brief with Claude (`langchain-anthropic`).

### Shared state (`src/recon/state.py`)

| Field | Purpose |
|---|---|
| `history` | Message log; appended to via the `add_messages` reducer |
| `competitors` | Competitors being tracked |
| `articles` | News items fetched for those competitors |
| `retrieved_docs` | Documents retrieved from Chroma |
| `brief` | The final sales brief |
| `next_agent` | Routing hint for the next node to run |

## Setup

Requires Python 3.13+ and [uv](https://docs.astral.sh/uv/).

```sh
uv sync
cp .env.example .env   # then fill in your keys
uv run python -m recon.main
```

## Development

```sh
uv run pytest
```

Evaluation dependencies (`ragas`) live in the `eval` group: `uv sync --group eval`.

## Layout

```
src/recon/
  agents/    agent nodes
  config/    settings
  models/    data models (e.g. Article)
  state.py   shared graph state
  main.py    entrypoint
```

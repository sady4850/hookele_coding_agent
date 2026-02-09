# Hookele

**Hookele** is an autonomous coding agent designed for the [Terminal-Bench 2.0](https://github.com/alexgshaw/terminal-bench-2-leaderboard) benchmarking environment.

## Overview

Hookele is a lightweight, agentic system that solves programming tasks through iterative planning and tool execution. It prioritizes simplicity over complex heuristics, letting the language model drive decision-making.

## Architecture

Hookele uses a **two-phase approach**:

1. **Skill Classification** - A fast, inexpensive model analyzes the task and selects relevant domain-specific "skills" (curated knowledge snippets) to inject into the system prompt.

2. **Execution Loop** - The main model creates a plan, then iteratively executes tools until the task is complete or the iteration limit is reached.

## Execution Model

- Creates an initial plan in the first response
- Iteratively calls tools to make progress
- Learns from errors through feedback loops
- Automatically nudges the model when stuck or when commands fail
- Tracks plan updates and revisions throughout execution

## Tools

| Tool | Purpose |
|------|---------|
| `run_command` | Execute shell commands |
| `apply_patch` | Edit files using V4A patch format |
| `web_search` | Search the web for information |
| `search_docs` / `get_docs` | Look up library documentation |
| `update_plan` | Revise the execution plan |
| `task_complete` | Signal successful completion |

## Design Philosophy

- **Simplicity first** - Generic prompting without domain-specific heuristics
- **LLM-driven verification** - The model decides when and how to verify its work
- **Error-driven learning** - Failed commands inform the next attempt
- **Minimal scaffolding** - No regex extraction or auto-compile magic

## Model

Uses OpenAI's GPT-5.1 Codex models with configurable reasoning effort levels.

---
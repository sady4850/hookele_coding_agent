# Hookele

Autonomous AI agent for solving programming tasks in the Harbor terminal benchmarking environment (Terminal-Bench 2.0).

## Project Structure

```
hookele/
├── adapters/harbor/       # Harbor integration
│   ├── agent.py           # Main HookeleAgent class (v1.0.0)
│   ├── tools.py           # Tool implementations
│   └── utils.py           # EventLogger, utilities
├── hookele/core/          # Core logic (placeholder)
├── docs/                   # Architecture docs
└── jobs/                   # Job execution artifacts
```

## How It Works

**Two-phase approach:**

1. **Skill Classification** (gpt-4o-mini)
   - Analyzes task instruction with cheap model
   - Selects relevant skills from `skills/` directory
   - Injects skill content into system prompt

2. **Execution Phase** (gpt-5.1-codex-max)
   - Creates plan in first response via `update_plan` tool
   - Calls tools: `run_command`, `apply_patch`, `web_search`, etc.
   - Loops until `task_complete` or max iterations

## Running

```bash
~/.local/bin/harbor run \
    -d terminal-bench@2.0 \
    -t <task-name> \
    --agent-import-path "adapters.harbor.agent:HookeleAgent" \
    --jobs-dir ./jobs \
    --debug
```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `HOOKELE_MODEL` | `gpt-5.1-codex-mini` | Main execution model |
| `HOOKELE_SKILL_MODEL` | `gpt-5.1-codex-mini` | Model for skill classification |
| `HOOKELE_REASONING_EFFORT` | `medium` | Reasoning effort: none, low, medium, high |
| `HOOKELE_MAX_OUTPUT_TOKENS` | `12000` | Max output tokens per model turn |
| `HOOKELE_MAX_TOOL_OUTPUT_CHARS` | `20000` | Max characters returned by `run_command` output |
| `HOOKELE_API_BASE` | - | Custom API base URL |
| `HOOKELE_API_KEY` | `OPENAI_API_KEY` | API key |
| `HOOKELE_MAX_ITERATIONS` | `60` | Max tool-call iterations |
| `HOOKELE_STREAM_MAX_RETRIES` | `5` | Retries for dropped streaming responses |

## Job Artifacts

Each run creates timestamped directories in `jobs/`:
- `traj.jsonl` - Detailed event trajectory
- `trajectory.json` - ATIF-format trajectory file
- `summary.json` - Status, iterations, duration
- `verifier/reward.txt` - Final reward (0 or 1)

## Trajectory Validation

Harbor provides a trajectory validator for validating ATIF trajectory files. The validator checks that trajectory files conform to the ATIF schema.

### Usage

```bash
# Validate a single trajectory file (using Harbor's Python environment)
~/.local/share/uv/tools/harbor/bin/python -m harbor.utils.trajectory_validator trajectory.json

# Validate a trajectory from a job run
~/.local/share/uv/tools/harbor/bin/python -m harbor.utils.trajectory_validator jobs/2026-01-15__11-25-43/build-pov-ray__bwBsCu9/agent/trajectory.json

# Or if Harbor is installed via pip in your current environment:
python -m harbor.utils.trajectory_validator trajectory.json
```

**Example output:**
```
✓ Trajectory is valid: trajectory.json
```

### What it validates

The validator checks:
- Schema version compatibility (ATIF-v1.5)
- Required fields (session_id, agent, steps)
- Step structure and timestamps
- Agent metadata (name, version, model_name)
- Final metrics if present

This is useful for:
- Debugging trajectory generation issues
- Validating files before submission
- Ensuring compatibility with Harbor's trajectory processing tools

## Design Principles

- Keep it simple - generic prompt, no domain-specific heuristics
- LLM decides when to verify (via `run_command`)
- LLM learns from errors through iterative loop
- No regex extraction, no auto-compile magic


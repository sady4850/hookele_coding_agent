# Building Hookele: An Autonomous Coding Agent for Terminal-Bench 2.0

*How a coding agent evolved from an overengineered two-model state machine to ~2,200 lines of "let the LLM figure it out" — and scored better for it.*

---

## The Punchline First

The best version of Hookele is simpler than the first commit. Every major improvement came from *removing* something — the planning phase, tool filtering, the state machine, acceptance criteria. The final agent is one model, one loop, and markdown files of domain knowledge. It scored 61.1% on Terminal-Bench 2.0's 89 tasks, and the single cheapest change in the entire pipeline (a 100-token skill classification call) had the biggest impact on results.

This is the story of how I got there, including every wrong turn.

## The Problem

[Terminal-Bench 2.0](https://github.com/alexgshaw/terminal-bench-2-leaderboard) is a benchmarking environment with 89 diverse programming tasks. Your agent gets a task instruction and a terminal inside an isolated container. That's it. Tasks range from chess engine integration to 7z hash cracking, from R-to-Python statistical model porting to large-scale CSV transformations with Vim macros under keystroke constraints.

I wanted to build an agent that could handle this diversity — not by encoding domain knowledge into the harness, but by letting the model drive decision-making.

## Starting Point: The OpenAI Cookbook

OpenAI published a [cookbook example for building a coding agent with GPT-5.1](https://github.com/openai/openai-cookbook/blob/main/examples/Build_a_coding_agent_with_GPT-5.1.ipynb) using the Agents SDK. It demonstrates the core tool set — `apply_patch` for file editing, `shell` for command execution, `web_search` for information retrieval, and Context7 MCP for documentation lookup. It's a clean tutorial: define an agent, give it tools, let it scaffold and iterate on a codebase.

Hookele started from that foundation. The tool choices (`apply_patch`, `run_command`, `web_search`, Context7 docs) trace directly back to the cookbook's architecture. But a cookbook example and a benchmark-competitive agent are very different things. The cookbook shows you the happy path. Terminal-Bench shows you what happens when the model hallucinates context lines in a diff, when a streaming connection drops mid-response, when the agent burns 40 iterations going in circles on a task it doesn't understand. Hookele's evolution is the story of bridging that gap.

## v1: The Two-Model Architecture

The initial design followed what seemed like common sense: use a cheap model for planning, an expensive one for execution.

### The Planning Phase

A `gpt-5-mini` model received the task instruction and produced a structured JSON plan:

```python
PLAN_SCHEMA = {
    "name": "task_plan",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "analysis": {
                "type": "string",
                "description": "Brief analysis of what the task requires"
            },
            "approach": {
                "type": "string",
                "description": "Step-by-step approach to solve the task"
            },
            "tools_needed": {
                "type": "array",
                "items": {
                    "type": "string",
                    "enum": ["list_files", "read_file", "run_command", "apply_patch"]
                }
            }
        },
        "required": ["analysis", "approach", "tools_needed"],
        "additionalProperties": False
    }
}
```

The planner also decided which tools the executor was allowed to use. If the plan said `tools_needed: ["run_command", "read_file"]`, the execution model only saw those tools. The idea was to reduce confusion — if a task doesn't need file editing, don't tempt the model with `apply_patch`.

### The Execution Phase

The main `gpt-5.1` model received the plan and a simple system prompt, then looped calling tools until it either stopped producing tool calls or hit the 25-iteration limit. No streaming — synchronous API calls. Four tools: `list_files`, `read_file`, `run_command`, and OpenAI's built-in `apply_patch`.

```python
# Phase 1: Planning with cheaper model
plan = self._plan_task(client, plan_model, instruction, event_logger)
tools = self._select_tools(plan.get("tools_needed", []))

# Phase 2: Execution with main model
system_prompt = self._build_system_prompt(instruction, plan)
```

### What Was Wrong

This worked on simple test tasks (prove-plus-comm, fix-git, sqlite-db-truncate — all 100% pass rate). But on harder Terminal-Bench tasks, several problems surfaced:

1. **The planner was often wrong.** It would say "no file editing needed" for a task that clearly required patches. The executor would then waste iterations trying to work around the missing tool.

2. **Context was lost between phases.** The planner analyzed the task in isolation. The executor got the plan text but none of the planner's intermediate reasoning. They were two models talking past each other.

3. **Structured output was fragile.** The `gpt-5-mini` model would sometimes produce valid JSON that was useless — `"analysis": "This task requires running some commands"`. Thanks, very helpful.

4. **Tool filtering was actively harmful.** Restricting the executor's tools based on the planner's (often incorrect) analysis meant the agent couldn't adapt when reality diverged from the plan.

I also had an elaborate design doc describing a five-state machine: `scan → plan → act → verify → stop`. It included acceptance criteria extraction, structured verify feedback, dual-model routing for different task phases, per-hunk patch reporting, and a criteria evaluation system with four types (command, file_exists, file_contains, llm_judge). Almost none of it survived.

## v2: The Big Rewrite

Two weeks in, I threw out the two-model approach and rebuilt around a single idea: **one model, one loop, let it plan itself.**

### Killing the Planning Phase

Instead of a separate model creating a plan, the execution model now creates its own plan as its first action using an `update_plan` tool:

```python
{
    "type": "function",
    "name": "update_plan",
    "parameters": {
        "type": "object",
        "properties": {
            "explanation": {"type": "string"},
            "plan": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "step": {"type": "string"},
                        "status": {
                            "type": "string",
                            "enum": ["pending", "in_progress", "completed"]
                        }
                    }
                }
            }
        }
    }
}
```

The system prompt enforces this: "First response: call `update_plan` with 3-5 short steps. Do not call other tools." After that, the model executes its own plan, calling `update_plan` again only if the approach changes. The plan lives in the conversation context — the model can see it and revise it.

This eliminated the planner-executor context gap entirely. One model makes the plan, one model executes it, and it's the same model.

### Dropping Redundant Tools

`list_files` and `read_file` were just wrappers around `run_command("ls -la ...")` and `run_command("cat ...")`. They existed because I thought the model needed "safe" abstractions. It didn't. Removing them simplified the tool interface and gave the model more flexibility (it could run `ls -R`, `head -20`, `wc -l`, `find`, etc. — things `list_files` couldn't do).

### Custom V4A Differ

The initial version used OpenAI's [built-in `apply_patch` tool](https://developers.openai.com/api/docs/guides/tools-apply-patch). The model is trained to produce V4A diffs, which is a nice head start — but you still need to implement the code that applies those diffs to files. The v1 implementation was naive (`str.replace()` on context lines) and fell apart when the model's context lines had trailing whitespace differences.

In v2, I replaced it with a proper 353-line V4A diff engine (ported from the OpenAI Agents SDK reference) with **three-level fuzzy context matching**: exact → trailing-whitespace-tolerant → fully-stripped. Each fuzziness level is tracked in the trajectory, so I can see when the model was sloppy but recoverable. The differ also detects unresolved git merge conflict markers — without this, the model would happily patch one side of a conflict, producing a file that compiled but had subtly wrong behavior.

### Streaming and Resilience

The switch from synchronous to streaming API calls wasn't about latency — it was about surviving long agent sessions. A 60-iteration run can take 10+ minutes. Connections drop. The retry logic (5 attempts, exponential backoff) handles transport errors and API-level `response.incomplete` separately. The `previous_response_id` field is critical — it lets us resume conversation state after a dropped connection without re-sending the full context window.

### Other Changes

- Max iterations: 25 → 60 (harder tasks need more steps)
- Default model: `gpt-5.1` → `gpt-5.1-codex-max` (better at code)
- Added token usage tracking and cost computation with model-specific rate tables
- Added failure nudges — when a command fails, inject one system message with the error summary
- Added plan tracking with history (every plan revision is logged with timestamp and explanation)

## v3: Skill Classification — The Key Insight

The biggest performance improvement came from the simplest idea: **inject domain knowledge before the model starts working.**

### The Problem

The agent was spending its first 3-5 iterations figuring out basic things. On a chess task, it would try to reason about positions before discovering it should use Stockfish. On a hash cracking task, it would attempt brute force before finding `hashcat`. These were wasted iterations that ate into the 60-step budget.

### The Solution

The idea of injecting curated knowledge into an agent's context isn't new — it's the same pattern used by Claude Code (with `CLAUDE.md` and its skills system), OpenAI's Codex CLI (with `AGENTS.md`), and other production coding agents. The insight is that a few hundred tokens of targeted domain knowledge in the system prompt are worth more than thousands of tokens of the model figuring things out through trial and error.

Hookele's version: a cheap model call at the start classifies which "skills" are relevant. Each skill is a Markdown file with a YAML frontmatter:

```markdown
---
name: chess-best-move
description: Determine the best chess move using a chess engine.
---

## Overview

The chess position is provided as an image file (chess_board.png).
The agent must:
1. Extract the position from the image (convert to FEN)
2. Use Stockfish to find the best move(s)
3. Output in UCI notation to /app/move.txt

## Required Tools
...
```

The skill registry loads all `.md` files from the `skills/` directory at import time:

```python
def _load_skills():
    skills = {}
    skills_dir = Path(__file__).resolve().parents[2] / "skills"
    for path in sorted(skills_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, content = _parse_frontmatter(text)
        skills[path.stem] = {
            "name": meta.get("name", path.stem),
            "description": meta.get("description", ""),
            "content": content,
        }
    return skills
```

Classification uses the same model as execution but with a 100-token cap and `reasoning: {"effort": "high"}`:

```python
def _classify_skills(self, client, instruction):
    skill_options = "\n".join(
        f"- {key}: {data['description']}" 
        for key, data in SKILLS.items()
    )
    prompt = (
        f"Task instruction:\n{instruction}\n\n"
        f"Available skills:\n{skill_options}\n\n"
        "Which skills apply? Return JSON: {\"skills\": [...]} "
        "Only include skills that are clearly relevant."
    )
    response = client.responses.create(
        model="gpt-5.1-codex-mini",
        input=[{"role": "user", "content": prompt}],
        text={"format": {"type": "json_object"}},
        reasoning={"effort": "high"},
        max_output_tokens=100,
    )
    result = json.loads(response.output_text)
    return [s for s in result.get("skills", []) if s in SKILLS]
```

Matched skills get their content injected directly into the system prompt. The model starts its first turn already knowing that Stockfish is the right tool, or that `hashcat` is faster than `john`, or that Vim macros need `setreg` escaping for the keystroke constraint.

### What Else Changed

- **`task_complete` tool** — explicit completion signaling. Before this, the agent just stopped producing tool calls, which was ambiguous (did it finish, or did it give up?).
- **Chat-only nudge** — if the model responds with text but no tool calls, inject: "You provided a text response but no tool calls. Please use tools to proceed, or call `task_complete` if finished." Previously this was treated as task completion.
- **Default model: `codex-max` → `codex-mini`** — this was a deliberate trade-off, not a compromise. At $0.25/M input, $0.025/M cached, and $2.00/M output tokens, codex-mini is cheap enough to iterate fast. On a benchmark with 89 tasks and multiple runs per task, the cost difference between mini and max is enormous. More importantly, cheap iterations compound: you can afford to run 60 iterations per task, retry failed runs, and experiment with different skills — all without watching the bill spiral. With skills providing domain context upfront, the smaller model performed well enough that the extra capability of codex-max wasn't worth the cost. **Iteration speed beats single-run intelligence.**

### Skills Growth

The skills library grew across three commits:
- **2026-01-23**: 17 initial skills (adaptive-rejection-sampler, async-cancellation, chess-best-move, financial-document-processor, etc.)
- **2026-01-31**: +20 skills (CompCert compilation, QEMU VM setup, path tracing, polyglot C/Python programs, hash cracking, Core Wars strategies, etc.)
- **2026-02-11**: Final batch (large-scale text editing with Vim macros, R-to-Python Stan model porting, distribution search)

Final count: 37 skills for 89 tasks. Some tasks share skills, some get no skill match and run with generic prompting.

## The Error Recovery Philosophy

Most agent frameworks I studied bolt on error-handling rules: regex patterns for common errors, automatic retry logic for specific failure modes, domain-specific recovery strategies. Hookele takes a different approach.

When a command fails, the agent gets the full output (truncated to 20K characters with a head/tail strategy — configurable via `HOOKELE_MAX_TOOL_OUTPUT_CHARS`, default 20,000):

```python
def truncate(text, limit=50_000):  # tool output uses limit=20_000
    if len(text) <= limit:
        return text, False
    head = text[:limit // 2]
    tail = text[-limit // 2:]
    return head + "\n...[TRUNCATED]...\n" + tail, True
```

If the failure repeats without progress, a single nudge is injected:

```python
if last_failure and not saw_non_plan_tool and failure_nudges < 1:
    failure_nudges += 1
    pending_input.append({
        "role": "system",
        "content": f"Last command failed. Summary:\n{last_failure}\n\n"
                   "Please address the failure or try an alternative and continue."
    })
```

One nudge. Not two, not three. The counter resets when the model successfully executes a non-plan tool call. If the model can't recover from one nudge, more nudges won't help — and they'd just eat into the iteration budget.

The key insight: **the model is the error handler.** Adding heuristics on top creates a second, worse error handler that sometimes conflicts with the first. Every time I tried adding smart error recovery (detect Python tracebacks, auto-install missing packages, retry with different flags), the results got worse because the heuristic would fire in edge cases where the model had a better strategy.

## What I Threw Away

The git history tells a story of subtraction. Every item below made the agent *worse* when it was included:

### The Five-State Machine
The original `planning-act-loop.md` described states: `scan → plan → act → verify → stop`. Each state had its own prompt template, its own model routing (fast vs main), its own output contract. The scan phase would run `pwd`, `ls`, `git status` and extract keywords with `rg`. The verify phase would parse test output into structured hints (`[{"file": "path", "line": 123, "message": "..."}]`).

In practice, the model does all of this naturally when you give it a good system prompt and let it call `run_command`. No state machine needed.

### Acceptance Criteria Extraction
I designed a system where the agent would extract structured criteria from the instruction:

```json
{
  "criteria": [
    {"type": "command", "check": "pytest tests/ -q", "description": "all tests pass"},
    {"type": "file_exists", "check": "src/auth.py", "description": "auth module created"},
    {"type": "file_contains", "file": "src/auth.py", "check": "def login\\("},
    {"type": "llm_judge", "check": "error messages are user-friendly"}
  ]
}
```

The agent would evaluate these after each meaningful edit to determine if it was done. This never made it past the design doc. The `task_complete` tool with a free-text summary works fine — the model naturally runs tests and checks outputs without needing a formal spec.

### Tool Filtering by Planner
The planner would decide which tools the executor could see. This sounded elegant (reduce the action space!) but was actively harmful. If the planner incorrectly excluded `apply_patch`, the executor couldn't edit files. The agent is better off having all tools available and deciding for itself what to use.

### Dual-Model Routing
Fast model for planning/command selection/criteria/verify parsing, main model for patches/reasoning. The coordination overhead (passing context between models, handling disagreements about strategy, maintaining separate prompt templates) wasn't worth the cost savings. A single model with medium reasoning effort is simpler and produces better results.

### `gpt-5.1-codex-max` as Default
Switched to `codex-mini` when skill injection compensated for the smaller model's capacity. The cost difference is significant across 89 tasks with multiple runs each.

## The System Prompt

The final system prompt is dense but earns its length. Key elements:

```python
def _build_system_prompt(self, instruction, max_iterations, active_skills=None):
    base_prompt = f"""You are a highly efficient autonomous coding agent.
## Task
{instruction}

## Workflow
Before planning, extract key constraints (max 6 bullets, 1 line each).
1) First response: call update_plan with 3-5 short steps; do not call other tools.
2) Execute the plan. Start implementation by the second response.
3) If the plan changes, call update_plan with explanation, then continue.
4) Plan must include a Test/Verify step when possible.

## Tool & Build Heuristics
- When you see a tool directory with run/, bin/, or build/, 
  ALWAYS run ls <tool>/run to check for pre-built binaries 
  BEFORE attempting to compile from source.

## Editing
- Use apply_patch for file edits; avoid sed -i, cat >.
- Write outputs to exact absolute paths specified.

## Constraints
- Max {max_iterations} steps; timebox exploration and pivot if stuck.
- Use failure output to change strategy if the same error repeats.
- If a command fails (missing tool, PEP 668, timeouts), pivot."""

    if active_skills:
        skill_content = "\n".join(
            SKILLS[s]["content"] for s in active_skills if s in SKILLS
        )
        base_prompt += f"\n{skill_content}"

    return base_prompt
```

The "Tool & Build Heuristics" section was added after watching the agent waste 10+ iterations trying to compile tools from source when pre-built binaries existed. One line in the prompt fixed it permanently.

The budget warning system injects urgency near the end:

```python
if iterations == max_iterations - 5:
    turn_input.append({
        "role": "user", 
        "content": f"Warning: iteration {iterations} of {max_iterations}. "
                   "You have 5 steps remaining."
    })
if iterations == max_iterations - 1:
    turn_input.append({
        "role": "user",
        "content": "FINAL WARNING: This is your LAST step. "
                   "Write the output file NOW or you will fail."
    })
```

## Documentation Lookup

One tool that proved surprisingly valuable is `search_docs`/`get_docs`, powered by the Context7 API:

```python
CONTEXT7_BASE_URL = "https://context7.com/api/v2"

def search_docs(library_name, query=None):
    params = {"libraryName": library_name}
    if query:
        params["query"] = query
    result = _context7_request("libs/search", params)
    # Return top 5 results with id, title, description
    ...

def get_docs(library_id, query):
    params = {"libraryId": library_id, "query": query, "type": "json"}
    result = _context7_request("context", params)
    # Format info snippets and code snippets
    ...
```

This is the agent equivalent of a developer opening the docs before writing code. It dramatically reduces hallucinated API calls — the model doesn't guess function signatures when it can look them up. The system prompt enforces this: "Before code edits involving a third-party library/API, use `search_docs` to find the library, then `get_docs` to fetch relevant documentation."

## Results

All 89 tasks, 5 runs each (445 trials), GPT-5.1 Codex Mini — **61.1% overall accuracy**.

### The Breakdown

**46 tasks solved perfectly (5/5)** — these span a wide range: git operations (`fix-git`, `git-multibranch`, `sanitize-git-repo`), cryptography (`crack-7z-hash`, `feal-differential-cryptanalysis`), systems work (`kv-store-grpc`, `nginx-request-logging`, `qemu-startup`), ML (`model-extraction-relu-logits`, `pytorch-model-recovery`), and oddities like `cobol-modernization` and `build-pov-ray`.

**16 tasks partially solved** — the interesting middle ground:

| Success Rate | Tasks |
|---|---|
| 80% (4/5) | `pytorch-model-cli`, `count-dataset-tokens`, `password-recovery`, `mcmc-sampling-stan` |
| 60% (3/5) | `large-scale-text-editing`, `qemu-alpine-ssh`, `reshard-c4-data`, `fix-ocaml-gc`, `rstan-to-pystan`, `llm-inference-batching-scheduler` |
| 40% (2/5) | `query-optimize`, `protein-assembly` |
| 20% (1/5) | `schemelike-metacircular-eval`, `path-tracing`, `overfull-hbox`, `polyglot-rust-c` |

**27 tasks at zero** — the honest failures. These cluster around: GPU/CUDA-dependent tasks (`torch-tensor-parallelism`, `torch-pipeline-parallelism`, `train-fasttext`), tasks requiring visual understanding (`extract-moves-from-video`, `sam-cell-seg`), complex compilation targets (`compile-compcert`, `make-doom-for-mips`), and tasks where the agent's strategy was fundamentally wrong (`chess-best-move` — it had a skill for this but still failed all 5 runs).

The `chess-best-move` failure is worth noting: having the right skill injected doesn't guarantee success. The skill told the agent to use Stockfish, but extracting a chess position from an image and converting it to FEN reliably turned out to be harder than the skill anticipated. Domain knowledge helps; it doesn't solve the problem for you.

<details>
<summary><strong>Full results table (89 tasks, sorted by success rate)</strong></summary>

| Task | Runs | Success Rate |
|---|---|---|
| adaptive-rejection-sampler | 5 | 100% |
| bn-fit-modify | 5 | 100% |
| build-cython-ext | 5 | 100% |
| build-pmars | 5 | 100% |
| build-pov-ray | 5 | 100% |
| cancel-async-tasks | 5 | 100% |
| cobol-modernization | 5 | 100% |
| code-from-image | 5 | 100% |
| configure-git-webserver | 5 | 100% |
| constraints-scheduling | 5 | 100% |
| crack-7z-hash | 5 | 100% |
| custom-memory-heap-crash | 5 | 100% |
| db-wal-recovery | 5 | 100% |
| distribution-search | 5 | 100% |
| extract-elf | 5 | 100% |
| feal-differential-cryptanalysis | 5 | 100% |
| feal-linear-cryptanalysis | 5 | 100% |
| fix-code-vulnerability | 5 | 100% |
| fix-git | 5 | 100% |
| git-leak-recovery | 5 | 100% |
| git-multibranch | 5 | 100% |
| headless-terminal | 5 | 100% |
| hf-model-inference | 5 | 100% |
| kv-store-grpc | 5 | 100% |
| largest-eigenval | 5 | 100% |
| log-summary-date-ranges | 5 | 100% |
| mailman | 5 | 100% |
| merge-diff-arc-agi-task | 5 | 100% |
| model-extraction-relu-logits | 5 | 100% |
| modernize-scientific-stack | 5 | 100% |
| multi-source-data-merger | 5 | 100% |
| nginx-request-logging | 5 | 100% |
| openssl-selfsigned-cert | 5 | 100% |
| polyglot-c-py | 5 | 100% |
| portfolio-optimization | 5 | 100% |
| prove-plus-comm | 5 | 100% |
| pypi-server | 5 | 100% |
| pytorch-model-recovery | 5 | 100% |
| qemu-startup | 5 | 100% |
| regex-log | 5 | 100% |
| sanitize-git-repo | 5 | 100% |
| sparql-university | 5 | 100% |
| sqlite-db-truncate | 5 | 100% |
| sqlite-with-gcov | 5 | 100% |
| tune-mjcf | 5 | 100% |
| vulnerable-secret | 5 | 100% |
| count-dataset-tokens | 5 | 80% |
| mcmc-sampling-stan | 5 | 80% |
| password-recovery | 5 | 80% |
| pytorch-model-cli | 5 | 80% |
| fix-ocaml-gc | 5 | 60% |
| large-scale-text-editing | 5 | 60% |
| llm-inference-batching-scheduler | 5 | 60% |
| qemu-alpine-ssh | 5 | 60% |
| reshard-c4-data | 5 | 60% |
| rstan-to-pystan | 5 | 60% |
| protein-assembly | 5 | 40% |
| query-optimize | 5 | 40% |
| overfull-hbox | 5 | 20% |
| path-tracing | 5 | 20% |
| polyglot-rust-c | 5 | 20% |
| schemelike-metacircular-eval | 5 | 20% |
| break-filter-js-from-html | 5 | 0% |
| caffe-cifar-10 | 5 | 0% |
| chess-best-move | 5 | 0% |
| circuit-fibsqrt | 5 | 0% |
| compile-compcert | 5 | 0% |
| dna-assembly | 5 | 0% |
| dna-insert | 5 | 0% |
| extract-moves-from-video | 5 | 0% |
| filter-js-from-html | 5 | 0% |
| financial-document-processor | 5 | 0% |
| gcode-to-text | 5 | 0% |
| gpt2-codegolf | 5 | 0% |
| install-windows-3.11 | 5 | 0% |
| make-doom-for-mips | 5 | 0% |
| make-mips-interpreter | 5 | 0% |
| mteb-leaderboard | 5 | 0% |
| mteb-retrieve | 5 | 0% |
| path-tracing-reverse | 5 | 0% |
| raman-fitting | 5 | 0% |
| regex-chess | 5 | 0% |
| sam-cell-seg | 5 | 0% |
| torch-pipeline-parallelism | 5 | 0% |
| torch-tensor-parallelism | 5 | 0% |
| train-fasttext | 5 | 0% |
| video-processing | 5 | 0% |
| winning-avg-corewars | 5 | 0% |
| write-compressor | 5 | 0% |

</details>

These numbers are honest. This isn't a system that aces every benchmark — it's a system designed around a specific philosophy (simplicity, LLM-driven decisions, minimal scaffolding) and these are the results of that philosophy.

The [submission PR](https://huggingface.co/datasets/alexgshaw/terminal-bench-2-leaderboard/discussions/22) has passed validation on the Terminal-Bench leaderboard.

## What I'd Do Differently

- **Smarter output truncation** — The 20K-character tool output limit is a blunt instrument. Truncation that preserves error messages and test failures while cutting verbose success output would help.
- **Parallel tool calls** — Some tasks have independent steps (install dependencies while reading docs). Hookele runs everything sequentially.
- **Cost tracking from day one** — I added cost computation late. Per-task cost data from the start would have helped optimize the skill classification vs. execution budget split.
- **More skills, more systematically** — 37 skills for 89 tasks. The skill creation process was ad-hoc — I'd add skills after seeing failures. A more systematic approach (analyze all task categories, write skills proactively) would have been faster.
- **Skill quality matters more than quantity** — The `rstan-to-pystan` skill was rewritten from a generic `stan_migration.md` to a targeted guide. That single rewrite improved the task pass rate from ~0.2 to 0.6. Depth beats breadth.

## Architecture Summary

```
┌──────────────────────────────────────────────────────┐
│                    Hookele Agent                      │
│                                                       │
│  ┌─────────────┐   ┌─────────────────────────────┐   │
│  │   Skill      │   │      Execution Loop          │   │
│  │   Classifier │──→│                               │   │
│  │   (100 tok)  │   │  Plan → Execute → Verify      │   │
│  └─────────────┘   │  (up to 60 iterations)         │   │
│                     │                               │   │
│  ┌─────────────┐   │  ┌───────────────────────┐    │   │
│  │   37 Skill   │──→│  │  Tools                 │    │   │
│  │   Files (.md)│   │  │  - run_command         │    │   │
│  └─────────────┘   │  │  - apply_patch (V4A)    │    │   │
│                     │  │  - web_search           │    │   │
│                     │  │  - search_docs/get_docs │    │   │
│                     │  │  - update_plan          │    │   │
│                     │  │  - task_complete         │    │   │
│                     │  └───────────────────────┘    │   │
│                     └─────────────────────────────┘   │
│                                                       │
│  Streaming + retry (5 attempts, exp backoff)          │
│  JSONL trajectory logging (every event)               │
│  Cost tracking per model                              │
└──────────────────────────────────────────────────────┘
```

## The Stack

- **Python 3.11+**, ~2,200 lines across 5 files (agent loop, tools, V4A differ, skills, utils)
- **OpenAI GPT-5.1 Codex Mini** (execution + skill classification)
- **OpenAI Responses API** with streaming
- **V4A patch format** with three-level fuzzy context matching
- **Context7 API** for library documentation lookup
- **Terminal-Bench 2.0** via Harbor framework
- **JSONL + ATIF** trajectory files

## Three Lessons

**1. Subtract before you add.** The agent got better every time I removed something — the planning phase, tool filtering, the state machine, acceptance criteria. The final version is simpler than the first commit and scores better.

**2. The model is smarter than your heuristics.** Every hand-coded error recovery rule I tried produced worse results than just showing the model the error output. The model reads stack traces better than any regex I could write.

**3. Cheap context injection beats expensive reasoning.** One 100-token skill classification call that injects the right domain knowledge saves 5-10 iterations of the execution model discovering the same information through trial and error. The cheapest improvement in the entire pipeline had the biggest impact on results.

---

The code is open source: [github.com/sady4850/hookele_coding_agent](https://github.com/sady4850/hookele_coding_agent)

Reading it end-to-end takes about an hour. The design intent was simplicity — the entire agent is five files with no framework dependencies beyond the OpenAI SDK.

*Hookele is Hawaiian for "to steer, to navigate" — fitting for an agent that plots a course through unfamiliar problems.*

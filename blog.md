# Building Hookele: An Autonomous Coding Agent for Terminal-Bench 2.0

*+18 points over the baseline with the same model. Turns out the harness matters more than the model tier.*

---

## Why Build Hookele?

I wanted to answer a simple question: **how far can careful harness design push a merely-good model?** If model choice were everything, a mid-tier GPT-5.1 Codex Mini run should lag far behind GPT-5.1 Codex Max submissions. Hookele deliberately leans on a smaller model to see whether orchestration, tool UX, and domain context can close that gap.

## The Punchline

The final agent is one model, one loop, and a directory of skill files. **61.1%** across all 89 Terminal-Bench 2.0 tasks (445 trials) with GPT-5.1 Codex Mini — versus **43.1%** for the other codex-mini submission on the [official leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0?models=GPT-5.1-Codex-Mini). The full 5× sweep cost **$27** and took **35.4 hours**.

The largest single win was a 100-token skill-classification call that decides which Markdown skill sheets to stuff into the system prompt. Everything else is support structure so that one loop can keep moving: stable streaming, tool output capping at 20K characters, fuzzy V4A patches, Harbor-aware retries.

## Terminal-Bench in One Paragraph

[Terminal-Bench 2.0](https://github.com/alexgshaw/terminal-bench-2-leaderboard) drops your agent into a hardened Harbor container with only a task instruction and a terminal. Eighty-nine tasks (full registry: <https://www.tbench.ai/registry/terminal-bench/2.0>) span chess-engine guidance, R-to-Python Stan migrations, qemu bring-up, hash cracking, Core Wars, giant CSV surgery via Vim macros, and more. You get 60 iterations per run, no outside filesystem visibility, and no retries unless you build them yourself.

## Starting Point: The OpenAI Cookbook

I bootstrapped off OpenAI's [GPT-5.1 coding agent notebook](https://github.com/openai/openai-cookbook/blob/main/examples/Build_a_coding_agent_with_GPT-5.1.ipynb): `run_command`, `apply_patch`, `web_search`, Context7 documentation lookup. The cookbook shows the happy path; Harbor shows you what happens when diff context drifts, when a WebSocket drops mid-run, or when the model stalls for 40 iterations.

## v1: Two Models, Structured Optimism

**Architecture.** `gpt-5-mini` planned, `gpt-5.1` executed. The planner produced JSON (`analysis`, `approach`, `tools_needed`) and constrained the executor's toolset. The executor looped up to 25 times with `list_files`, `read_file`, `run_command`, and OpenAI's hosted `apply_patch`.

**Reality.**
- The planner frequently misclassified tasks ("no file editing needed") which meant the executor literally couldn't fix files.
- Structured JSON looked nice but contributed nothing the execution model couldn't infer on its own.
- Tool filtering handcuffed the agent the moment plan and reality diverged.
- Context split across two models added latency and lost nuance.

I also had a five-state machine design (scan → plan → act → verify → stop), acceptance criteria extraction, automated criteria evaluators, and dual-model routing. Almost none of it survived testing.

## v2: One Model, One Loop

**Kill the planner.** The executor now plans for itself via an `update_plan` tool. First response must be a 3–8 step plan; revisions happen only when the approach genuinely changes. The plan lives in the context window, so the model can inspect or mutate it at will.

**Drop redundant tools.** `list_files` and `read_file` disappeared. `run_command` already covers `ls`, `find`, `head`, `rg`, etc. Tool outputs are truncated to 20K characters (head + tail) so Harbor doesn't drown the model in log spam.

**V4A differ that actually works.** I ported the Agents SDK V4A engine (353 lines) and added three-level fuzzy context matching (exact → strip-trailing → fully-stripped) plus merge-conflict detection. That single change killed most patch drift.

**Streaming + Harbor resilience.** Streaming isn't about latency; it's about surviving Harbor's long-lived WebSocket sessions. Hookele retries five times on transport errors, remembers `previous_response_id`, and resumes seamlessly after 60-second hiccups. When Harbor restarts the pseudo-TTY mid-run (it happens), the agent just reconnects and continues.

## Tools That Actually Matter

After ripping out the overbuilt state machine, the agent settled on five essentials:

| Tool | Purpose |
|---|---|
| `run_command` | Swiss-army knife for shell, `ls`, `head`, `pytest`, everything. |
| `apply_patch` (V4A) | All edits go through a single, fuzz-tolerant differ. |
| `update_plan` | Keeps the model honest about strategy shifts and iteration budget. |
| `search_docs` / `get_docs` | Context7 lookup to avoid hallucinating library APIs. |
| `web_search` | Rarely used, but critical when Harbor tasks require fresh external knowledge (e.g., spec lookups). |
| `task_complete` | Explicit stopping condition with a human-readable summary. |

That’s it—everything else (`list_files`, `read_file`, bespoke “verify” tools) turned out to be noise. The fewer decisions the agent has to make about *which* tool to call, the more time it spends actually solving the task.

## v3: Skill Classification (The 100-Token Superpower)

Agents like Claude Code and the Codex CLI lean on skill files. Hookele does the same, but the routing is explicit: a 100-token codex-mini call reads the task instruction, evaluates the skill catalog, and returns JSON `{ "skills": [ ... ] }`. Each matched skill injects Markdown directly into the system prompt, so the execution model begins its very first turn with domain-specific heuristics.

```markdown
---
name: chess-best-move
description: Determine the best move from a board image using Stockfish.
---
Use `convert_board.py` to OCR `chess_board.png` into FEN, then call Stockfish with `stockfish fen <FEN> depth 18`. Output the best move in UCI format to `/app/move.txt`.
```

The classifier might return `{"skills": ["chess-best-move"]}`. The prompt then gains the block above verbatim. That single paragraph often saves 5–8 iterations of fumbling (e.g., trying to compute chess moves without ever touching Stockfish).

## Error Recovery, Harbor Edition

Hookele doesn't regex-match stack traces or auto-rerun commands. Instead:

- Every failed command's truncated output (20K cap) goes straight back to the model.
- If two consecutive steps fail without progress, I inject **one** nudge: "Last command failed. Summary: ... Try an alternative and continue." The counter resets on the next successful tool call.
- Harbor-specific issues—container restarts mid-stream, apt locks from parallel tasks, Context7 rate limits—get summarized and handed to the model. It's better at picking the next move than any heuristic I wrote.

The retry stack tracks two failure classes:
1. **Transport** (broken pipe, TLS handshake, `response.incomplete`) handled before the model ever sees them.
2. **Harbor runtime** (task container died, `apply_patch` sees `<<<<<<< HEAD`, `pip install` blocked by PEP 668). Those are surfaced verbatim so the model can pivot.

## What Got Deleted (and Why)

| Feature | Why it died |
|---|---|
| Planner-selected tools | Wrong 20% of the time, disastrous when wrong. |
| Acceptance criteria extraction | A free-text `task_complete` summary works; structured criteria added overhead with no reliability gain. |
| Dual-model routing | Passing context between models was more expensive than just letting codex-mini reason a bit longer. |
| `list_files` / `read_file` wrappers | Redundant with `run_command` + shell primitives. |
| Chat-only completion detection | Replaced with `task_complete` so I can tell "I'm done" from "I'm stuck." |

## The System Prompt Snapshot

```text
You are a highly efficient autonomous coding agent.
Before planning, list up to six key constraints.
1) First response: call update_plan with 3-5 steps. No other tools.
2) Execute immediately; implementation must start by the second turn.
3) Revise the plan only when strategy changes.
4) Always include a Test/Verify step when feasible.
5) When you see a tool dir with run/ or bin/, run ls <tool>/run before compiling.
6) Edits must use apply_patch; don't rely on sed -i or cat >.
7) Max 60 iterations; warn yourself with 5 steps left.
```

Active skills get appended beneath those rules. The final two warnings ("5 steps left", "LAST step") are injected as user messages so the model feels the countdown.

## When Skills Weren't Enough (Chess)

`chess-best-move` has a detailed skill file, yet Hookele still went 0/5. The failure mode: OCR. Harbor provides a PNG board, but pytesseract misread pieces too often, so the downstream FEN→Stockfish chain started from garbage. Lesson: skills remove exploration tax, but they don't patch brittle toolchains. The fix would be a dedicated board parser, not more prompting.

## Results (and Why They Matter)

- **Total:** 89 tasks × 5 runs each → **61.1%** accuracy.
- **Perfect (5/5):** 46 tasks ranging from `kv-store-grpc` to `feal-differential-cryptanalysis` to `sanitize-git-repo`.
- **Partial (1–5/5):** 16 tasks; e.g., `pytorch-model-cli` at 80%, `large-scale-text-editing` at 60%, `path-tracing` at 20%.
- **Zeroes:** 27 tasks dominated by GPU/vision-heavy requirements (`torch-tensor-parallelism`, `sam-cell-seg`, `extract-moves-from-video`) and tasks demanding deep multi-step reasoning (`compile-compcert`, `make-doom-for-mips`, `regex-chess`). Harness design can't compensate for everything — on tasks that require long chains of precise reasoning or debugging novel compilation targets, codex-mini hits a ceiling that no amount of skill injection will lift. A larger model would likely recover some of these.

Full per-task results live on the [official leaderboard](https://www.tbench.ai/leaderboard/terminal-bench/2.0) and in the Hookele repo's run artifacts (JSONL trajectories + CSV summaries). Compared to the other codex-mini entry (43.1%), Hookele gained **+18 points** purely through loop design and skill injection.

## Cost & Time

- **API spend:** $27 total, summed from `final_metrics.total_cost_usd` across 436 trajectory logs in the submission bundle. That's ~$0.06 per task-run on average.
- **Wall-clock:** ~35.4 hours for a 5× full-benchmark sweep (first to last task run in the bundle).
- **Per-task range:** simple tasks (e.g., `fix-git`) finish in under a minute and cost fractions of a cent; heavy tasks (`compile-compcert`, `qemu-alpine-ssh`) can burn 10+ minutes and $0.20+ each.

## Reproduce It Yourself

1. **Install Harbor + dataset.** Follow the official guide: <https://harborframework.com/docs/datasets/running-tbench>.
2. **Clone Hookele.** `git clone https://github.com/sady4850/hookele_coding_agent && cd hookele_coding_agent`.
3. **Install deps.** `uv venv && uv pip install -e .` (or plain `pip install -e .`).
4. **Set credentials.** Export `OPENAI_API_KEY`, plus `CONTEXT7_API_KEY` if you want documentation lookup.
5. **Run:** `python -m hookele run --tasks terminal-bench-2.0 --max-iterations 60 --skills skills`. Add `--save-trajectory out/` to archive JSONL logs.
6. **Validate:** Upload the Harbor run bundle to the Terminal-Bench submission form (details in the leaderboard repo).

## Three Lessons That Stuck

1. **Subtract before you add.** Every major score bump came from deleting scaffolding the model didn't need.
2. **LLMs are better error handlers than heuristics.** Show the model the failure, give it one nudge, get out of the way.
3. **Cheap context beats expensive reasoning.** A 100-token classifier + curated skills outperformed hours spent tweaking planner logic or upgrading models.

Code: <https://github.com/sady4850/hookele_coding_agent>

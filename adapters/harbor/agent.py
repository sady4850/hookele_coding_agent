"""Hookele Harbor agent - main orchestration."""

import asyncio
import json
import os
import sys
import time
import tomllib
import uuid
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional

from harbor.agents.base import BaseAgent
from harbor.environments.base import BaseEnvironment
from harbor.models.agent.context import AgentContext
from harbor.models.trajectories.agent import Agent
from harbor.models.trajectories.final_metrics import FinalMetrics
from harbor.models.trajectories.step import Step
from harbor.models.trajectories.trajectory import Trajectory
from openai import OpenAI

from .tools import TOOLS, execute_function, run_command
from .utils import EventLogger, iso_now, truncate
from .skills import SKILLS

DEFAULT_STREAM_MAX_RETRIES = 5
DEFAULT_STREAM_RETRY_BASE_DELAY_S = 1.0
DEFAULT_STREAM_RETRY_MAX_DELAY_S = 8.0


class HookeleAgent(BaseAgent):
    """
    Harbor agent using OpenAI Responses API for execution.
    """

    SUPPORTS_ATIF: bool = True

    @staticmethod
    def name() -> str:
        return "hookele"

    def version(self) -> str:
        """Read version from pyproject.toml."""
        try:
            # Try to find pyproject.toml relative to this file
            agent_file = Path(__file__)
            project_root = agent_file.parent.parent.parent
            pyproject_path = project_root / "pyproject.toml"
            if pyproject_path.exists():
                with pyproject_path.open("rb") as f:
                    data = tomllib.load(f)
                    return data.get("project", {}).get("version", "unknown")
        except Exception:
            pass
        # Fallback to hardcoded version if reading fails
        return "1.0.0"

    async def setup(self, environment: BaseEnvironment) -> None:
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        try:
            (self.logs_dir / "setup.log").write_text(f"setup_started {iso_now()}\n")
        except Exception:
            pass

    async def run(
        self, instruction: str, environment: BaseEnvironment, context: AgentContext
    ) -> None:
        started = time.monotonic()

        hookele_dir = self.logs_dir / ".hookele"
        event_logger = EventLogger(hookele_dir)

        steps: List[Step] = []
        steps.append(Step(step_id=1, timestamp=iso_now(), source="user", message=instruction))

        main_model = os.getenv("HOOKELE_MODEL", "gpt-5.1-codex-mini")
        max_output_tokens = self._get_max_output_tokens()
        reasoning_effort = self._get_reasoning_effort()
        api_key = os.getenv("HOOKELE_API_KEY") or os.getenv("OPENAI_API_KEY")
        base_url = os.getenv("HOOKELE_API_BASE") or None
        client_kwargs: Dict[str, Any] = {}
        if api_key:
            client_kwargs["api_key"] = api_key
        if base_url:
            client_kwargs["base_url"] = base_url
        client = OpenAI(**client_kwargs)

        hookele_version = self.version()
        event_logger.log_event({
            "type": "start",
            "instruction": instruction,
            "hookele_version": hookele_version,
            "model": main_model,
            "reasoning_effort": reasoning_effort,
        })

        try:
            max_iterations = int(os.getenv("HOOKELE_MAX_ITERATIONS", "60"))
        except ValueError:
            max_iterations = 60
        stream_max_retries = self._get_stream_max_retries()
        # Select tools
        tools = TOOLS

        # Classify skills upfront with cheap model
        active_skills = self._classify_skills(client, instruction)
        if active_skills:
            event_logger.log_event({
                "type": "skills_classified",
                "active_skills": active_skills,
            })

        # Build system prompt with injected skills
        system_prompt = self._build_system_prompt(instruction, max_iterations, active_skills)
        previous_response_id: Optional[str] = None
        pending_input: List[Dict[str, Any]] = [{"role": "user", "content": instruction}]

        iterations = 0
        llm_calls = 0
        token_usage: Dict[str, int] = {}
        last_error: Optional[str] = None
        last_error_details: Optional[Dict[str, Any]] = None
        empty_response_retries = 0
        plan_only_retries = 0
        current_plan: Optional[str] = None
        current_plan_items: Optional[List[Dict[str, str]]] = None
        plan_history: List[Dict[str, Any]] = []
        last_failure: Optional[str] = None
        failure_nudges = 0

        def record_plan_update(
            plan_kind: str,
            plan_text: str,
            plan_items: Optional[List[Dict[str, str]]],
            explanation: Optional[str],
            iteration: int,
        ) -> None:
            nonlocal current_plan, current_plan_items, plan_history
            current_plan = plan_text
            current_plan_items = plan_items

            plan_history.append({
                "kind": plan_kind,
                "content": plan_text,
                "items": plan_items,
                "explanation": explanation,
                "iteration": iteration,
                "timestamp": iso_now(),
            })
            plan_preview, _ = truncate(plan_text, limit=2000)
            event_logger.log_event({
                "type": "plan_update",
                "kind": plan_kind,
                "iteration": iteration,
                "content": plan_preview,
                "items": plan_items,
                "explanation": explanation,
            })

        while iterations < max_iterations:
            iterations += 1
            llm_calls += 1

            event_logger.log_event({
                "type": "llm_call",
                "iteration": iterations,
                "model": main_model,
                "reasoning_effort": reasoning_effort,
            })

            turn_input = list(pending_input)
            pending_input = []

            # Warnings
            if iterations == max_iterations - 5:
                warn_text = f"Warning: You have reached iteration {iterations} of {max_iterations}. You have 5 steps remaining."
                turn_input.append({"role": "user", "content": warn_text})
                steps.append(Step(step_id=len(steps)+1, timestamp=iso_now(), source="system", message=warn_text))

            if iterations == max_iterations - 1:
                final_warn = "FINAL WARNING: This is your LAST step. Write the output file NOW or you will fail."
                turn_input.append({"role": "user", "content": final_warn})
                steps.append(Step(step_id=len(steps)+1, timestamp=iso_now(), source="system", message=final_warn))

            # Call LLM (streaming)
            request_payload: Dict[str, Any] = {
                "model": main_model,
                "instructions": system_prompt,
                "input": turn_input,
                "tools": tools,
                "tool_choice": "auto",
                "max_output_tokens": max_output_tokens,
            }
            if reasoning_effort:
                request_payload["reasoning"] = {"effort": reasoning_effort}
            if previous_response_id:
                request_payload["previous_response_id"] = previous_response_id

            stream_result: Optional[Dict[str, Any]] = None
            usage: Dict[str, int] = {}
            for attempt in range(stream_max_retries + 1):
                try:
                    stream_result = self._stream_response(
                        client,
                        request_payload,
                        event_logger,
                        iterations,
                    )
                except Exception as e:
                    error_str = str(e)
                    if self._is_retryable_stream_exception(error_str) and attempt < stream_max_retries:
                        delay_s = self._stream_retry_delay(attempt + 1)
                        event_logger.log_event({
                            "type": "stream_retry",
                            "iteration": iterations,
                            "attempt": attempt + 1,
                            "max_retries": stream_max_retries,
                            "error": error_str,
                            "delay_s": delay_s,
                        })
                        await asyncio.sleep(delay_s)
                        continue
                    last_error = error_str
                    event_logger.log_event({"type": "error", "error": last_error})
                    steps.append(Step(step_id=len(steps)+1, timestamp=iso_now(), source="system", message=f"Agent Error: {last_error}"))
                    stream_result = None
                    break

                usage = stream_result.get("usage") or {}
                if usage:
                    for key, value in usage.items():
                        if isinstance(value, int):
                            token_usage[key] = token_usage.get(key, 0) + value

                stream_error = stream_result.get("error")
                stream_error_details = stream_result.get("error_details")
                if stream_error and self._is_retryable_stream_error(stream_error, stream_error_details) and attempt < stream_max_retries:
                    delay_s = self._stream_retry_delay(attempt + 1)
                    event_logger.log_event({
                        "type": "stream_retry",
                        "iteration": iterations,
                        "attempt": attempt + 1,
                        "max_retries": stream_max_retries,
                        "error": stream_error,
                        "error_details": stream_error_details,
                        "delay_s": delay_s,
                    })
                    await asyncio.sleep(delay_s)
                    continue
                break

            if stream_result is None:
                break  # Stop on error

            if stream_result.get("error"):
                last_error = stream_result["error"]
                last_error_details = stream_result.get("error_details")
                error_event = {"type": "error", "error": last_error}
                if last_error_details:
                    error_event["details"] = last_error_details
                event_logger.log_event(error_event)
                error_message = f"Agent Error: {last_error}"
                if last_error_details:
                    error_message += f" details: {json.dumps(last_error_details, ensure_ascii=True)}"
                steps.append(Step(step_id=len(steps)+1, timestamp=iso_now(), source="system", message=error_message))
                break

            previous_response_id = stream_result.get("response_id") or previous_response_id
            content_text = stream_result.get("content_text", "")
            tool_calls = stream_result.get("tool_calls", [])
            builtin_tool_calls = stream_result.get("builtin_tool_calls", [])

            # Log raw response
            event_logger.log_event({
                "type": "raw_response",
                "iteration": iterations,
                "content": content_text,
                "usage": usage,
                "tool_calls": [
                    {"name": call.name, "arguments": call.arguments, "call_id": call.call_id}
                    for call in tool_calls
                ],
                "builtin_tool_calls": builtin_tool_calls,
            })
            plan_update_from_text = self._extract_plan_block(content_text)
            saw_plan_tool = False
            saw_non_plan_tool = False
            hit_task_complete = False
            iteration_failed = False
            iteration_failure_summary: Optional[str] = None
            
            step_msg = content_text
            if tool_calls:
                step_msg += "\n\nTool Calls:\n" + "\n".join(
                    [f"{t.name}({t.arguments})" for t in tool_calls]
                )
            
            if step_msg.strip():
                steps.append(Step(step_id=len(steps)+1, timestamp=iso_now(), source="agent", message=step_msg))

            # Handle Tool Calls
            if tool_calls:
                for tool_call in tool_calls:
                    func_name = tool_call.name
                    try:
                        args = json.loads(tool_call.arguments)
                    except json.JSONDecodeError:
                        args = {}

                    if func_name == "update_plan":
                        saw_plan_tool = True
                        plan_items = self._normalize_plan_items(args.get("plan"))
                        explanation = args.get("explanation")
                        plan_text = self._format_plan_items(plan_items)
                        if plan_text:
                            plan_kind = "plan" if not plan_history else "replan"
                            record_plan_update(
                                plan_kind,
                                plan_text,
                                plan_items,
                                explanation if isinstance(explanation, str) else None,
                                iterations,
                            )
                    elif func_name == "task_complete":
                        # Task completed successfully
                        # Log the summary and exit
                        event_logger.log_event({
                            "type": "early_exit",
                            "reason": "task_complete",
                            "iteration": iterations,
                            "summary": args.get("summary")
                        })
                        # Mark loop to break after tool execution loop
                        hit_task_complete = True
                    else:
                        saw_non_plan_tool = True

                    # Execute
                    result = await execute_function(environment, func_name, args)
                    result_output = result.get("output", "")
                    result_summary = result.get("summary")
                    tool_ok = result.get("ok", True)
                    if not tool_ok:
                        iteration_failed = True
                        iteration_failure_summary = result_summary or result_output
                    
                    event_logger.log_event({
                        "type": "tool_execution",
                        "name": func_name,
                        "call_id": tool_call.call_id,
                        "args": args,
                        "output": result_output,
                    })
                    
                    # Append tool output for next responses call
                    pending_input.append({
                        "type": "function_call_output",
                        "call_id": tool_call.call_id,
                        "output": result_output,
                    })

                    tool_message = result_output
                    steps.append(Step(step_id=len(steps)+1, timestamp=iso_now(), source="system", message=f"Tool Output ({func_name}): {tool_message[:1000]}"))

                if hit_task_complete:
                    break

                if iteration_failed:
                    last_failure = iteration_failure_summary or last_failure
                elif saw_non_plan_tool:
                    last_failure = None
                    failure_nudges = 0

                if plan_update_from_text and not saw_plan_tool:
                    plan_kind = plan_update_from_text["kind"]
                    plan_text = plan_update_from_text["content"]
                    if plan_text:
                        record_plan_update(plan_kind, plan_text, None, None, iterations)

                if last_failure and not saw_non_plan_tool and failure_nudges < 1:
                    failure_nudges += 1
                    event_logger.log_event({
                        "type": "failure_nudge",
                        "iteration": iterations,
                        "summary": last_failure[:5000],
                    })
                    pending_input.append({
                        "role": "system",
                        "content": f"Last command failed. Summary:\n{last_failure}\n\nPlease address the failure or try an alternative and continue."
                    })
                    continue

                if saw_plan_tool and not saw_non_plan_tool and plan_only_retries < 1 and not last_failure:
                    plan_only_retries += 1
                    event_logger.log_event({"type": "plan_only", "iteration": iterations})
                    pending_input.append({
                        "role": "system",
                        "content": "Plan recorded. Begin executing step 1 using the available tools."
                    })
                    continue
            else:
                # No tools called - LLM is done
                if content_text.strip():
                    if last_failure and failure_nudges < 1:
                        failure_nudges += 1
                        event_logger.log_event({
                            "type": "failure_nudge",
                            "iteration": iterations,
                            "summary": last_failure[:5000],
                        })
                        pending_input.append({
                            "role": "system",
                            "content": f"Last command failed. Summary:\n{last_failure}\n\nPlease address the failure or try an alternative and continue."
                        })
                        continue
                    if plan_update_from_text:
                        plan_kind = plan_update_from_text["kind"]
                        plan_text = plan_update_from_text["content"]
                        if plan_text:
                            record_plan_update(plan_kind, plan_text, None, None, iterations)
                    if plan_update_from_text and plan_only_retries < 1:
                        plan_only_retries += 1
                        event_logger.log_event({"type": "plan_only", "iteration": iterations})
                        pending_input.append({
                            "role": "system",
                            "content": "Plan recorded. Begin executing step 1 using the available tools."
                        })
                        continue
                    
                    # LLM responded with text but no actions - task complete?
                    # Nudge if it looks like it just forgot to call a tool
                    event_logger.log_event({"type": "chat_only_nudge", "iteration": iterations})
                    pending_input.append({
                        "role": "user",
                        "content": "You provided a text response but no tool calls. Please use tools to proceed with the task, or call 'task_complete' if you are finished."
                    })
                    continue
                else:
                    # Empty response - nudge once then exit
                    empty_response_retries += 1
                    if empty_response_retries <= 1:
                        if last_failure and failure_nudges < 1:
                            failure_nudges += 1
                            event_logger.log_event({
                                "type": "failure_nudge",
                                "iteration": iterations,
                                "summary": last_failure[:5000],
                            })
                            pending_input.append({
                                "role": "system",
                                "content": f"Last command failed. Summary:\n{last_failure}\n\nPlease address the failure or try an alternative and continue."
                            })
                        else:
                            pending_input.append({
                                "role": "user",
                                "content": "No output received. If you're done, confirm. Otherwise, continue with the next step."
                            })
                        continue
                    else:
                        event_logger.log_event({"type": "early_exit", "reason": "empty_responses", "iteration": iterations})
                        break

        hit_iteration_limit = iterations >= max_iterations
        if hit_iteration_limit:
            steps.append(Step(step_id=len(steps)+1, timestamp=iso_now(), source="system", message="Agent reached maximum iteration limit."))

        duration = round(time.monotonic() - started, 3)
        hookele_version = self.version()
        if last_error:
            reason = last_error
        elif hit_iteration_limit:
            reason = "agent reached maximum iteration limit"
        else:
            reason = "agent finished"
        summary = {
            "status": "completed",
            "exit_code": 0 if not last_error else 1,
            "reason": reason,
            "iterations": iterations,
            "llm_calls": llm_calls,
            "time_s": duration,
            "token_usage": token_usage,
            "error_details": last_error_details,
            "hookele_version": hookele_version,
            "model": main_model,
            "reasoning_effort": reasoning_effort,
            "plan": {
                "current": current_plan,
                "current_items": current_plan_items,
                "history": plan_history,
            },
        }
        event_logger.write_summary(summary)

        cost_usd = self._compute_cost_usd(token_usage, main_model)
        trajectory = self._build_trajectory(steps, token_usage, cost_usd, main_model)
        (self.logs_dir / "trajectory.json").write_text(json.dumps(trajectory.to_json_dict(), indent=2))

        context.metadata = {
            "status": summary["status"],
            "iterations": iterations,
            "duration_seconds": duration,
            "token_usage": token_usage,
        }
        if "input_tokens" in token_usage:
            context.metadata["n_input_tokens"] = token_usage["input_tokens"]
        if "output_tokens" in token_usage:
            context.metadata["n_output_tokens"] = token_usage["output_tokens"]
        if "cached_tokens" in token_usage:
            context.metadata["n_cache_tokens"] = token_usage["cached_tokens"]
        context.n_input_tokens = token_usage.get("input_tokens")
        context.n_output_tokens = token_usage.get("output_tokens")
        context.n_cache_tokens = token_usage.get("cached_tokens")
        context.cost_usd = cost_usd

    def _extract_plan_block(self, content: str) -> Optional[Dict[str, str]]:
        if not content:
            return None
        lines = content.splitlines()
        found: Optional[Dict[str, str]] = None
        i = 0
        while i < len(lines):
            stripped = lines[i].strip()
            lower = stripped.lower()
            if lower.startswith("plan:") or lower.startswith("replan:"):
                kind = "replan" if lower.startswith("replan:") else "plan"
                plan_lines: List[str] = []
                remainder = stripped.split(":", 1)[1].strip()
                if remainder:
                    plan_lines.append(remainder)
                i += 1
                while i < len(lines):
                    line = lines[i]
                    if not line.strip():
                        break
                    if line.strip().lower().startswith(("plan:", "replan:")):
                        break
                    plan_lines.append(line.rstrip())
                    i += 1
                plan_text = "\n".join(plan_lines).strip()
                if plan_text:
                    found = {"kind": kind, "content": plan_text}
                continue
            i += 1
        return found

    def _normalize_plan_items(self, raw_items: Any) -> List[Dict[str, str]]:
        if not isinstance(raw_items, list):
            return []
        normalized: List[Dict[str, str]] = []
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            step = str(item.get("step", "")).strip()
            status = str(item.get("status", "")).strip()
            if not step:
                continue
            if status not in {"pending", "in_progress", "completed"}:
                status = ""
            normalized.append({"step": step, "status": status})
        return normalized

    def _format_plan_items(self, items: List[Dict[str, str]]) -> str:
        lines: List[str] = []
        for item in items:
            step = item.get("step", "")
            status = item.get("status", "")
            if not step:
                continue
            if status:
                lines.append(f"- [{status}] {step}")
            else:
                lines.append(f"- {step}")
        return "\n".join(lines).strip()

    def _get_max_output_tokens(self) -> int:
        try:
            return int(os.getenv("HOOKELE_MAX_OUTPUT_TOKENS", "12000"))
        except ValueError:
            return 12000

    def _get_reasoning_effort(self) -> str:
        effort = os.getenv("HOOKELE_REASONING_EFFORT", "medium").strip().lower()
        if effort in {"none", "low", "medium", "high"}:
            return effort
        return "medium"

    def _classify_skills(self, client: OpenAI, instruction: str) -> List[str]:
        """Classify which skills apply to the task using a cheap model."""
        if not SKILLS:
            return []

        skill_model = os.getenv("HOOKELE_SKILL_MODEL", "gpt-5.1-codex-mini")
        skill_options = "\n".join(
            f"- {key}: {data['description']}" for key, data in SKILLS.items()
        )

        prompt = f"Task instruction:\n{instruction}\n\nAvailable skills:\n{skill_options}\n\nWhich skills apply to this task? Return JSON: {{\"skills\": [\"skill_key\", ...]}} or {{\"skills\": []}} if none apply. Only include skills that are clearly relevant."

        try:
            response = client.responses.create(
                model=skill_model,
                input=[{"role": "user", "content": prompt}],
                text={"format": {"type": "json_object"}},
                reasoning={"effort": "high"},
                max_output_tokens=100,
            )
            content = response.output_text or "{}"
            result = json.loads(content)
            skills = result.get("skills", [])
            # Validate skill keys
            return [s for s in skills if isinstance(s, str) and s in SKILLS]
        except Exception as e:
            # On any error, proceed without skills (but log it)
            print(f"[hookele] Skill classification failed: {e}", file=sys.stderr)
            return []

    def _get_stream_max_retries(self) -> int:
        raw = os.getenv("HOOKELE_STREAM_MAX_RETRIES")
        if raw is None:
            return DEFAULT_STREAM_MAX_RETRIES
        try:
            return max(0, int(raw))
        except ValueError:
            return DEFAULT_STREAM_MAX_RETRIES

    def _stream_retry_delay(self, attempt: int) -> float:
        delay = DEFAULT_STREAM_RETRY_BASE_DELAY_S * (2 ** max(attempt - 1, 0))
        return min(delay, DEFAULT_STREAM_RETRY_MAX_DELAY_S)

    def _is_retryable_stream_exception(self, error: str) -> bool:
        if not error:
            return False
        lowered = error.lower()
        retry_signals = (
            "incomplete chunked read",
            "peer closed connection",
            "stream closed",
            "connection reset",
            "connection aborted",
            "connection error",
            "broken pipe",
            "timed out",
            "timeout",
            "eof",
            "ssl",
            "tls",
            "record layer failure",
            "handshake",
        )
        return any(signal in lowered for signal in retry_signals)

    def _is_retryable_stream_error(
        self, error: str, error_details: Optional[Dict[str, Any]]
    ) -> bool:
        if not error:
            return False
        lowered = str(error).lower()
        reason = str((error_details or {}).get("reason", "")).lower()

        if "response.incomplete" in lowered:
            return any(token in reason for token in ("stream", "timeout", "connection"))
        if "response.failed" in lowered:
            return any(token in reason for token in ("stream", "timeout", "connection", "server", "rate"))
        if "stream" in lowered:
            return True
        return False

    def _stream_response(
        self,
        client: OpenAI,
        request_payload: Dict[str, Any],
        event_logger: EventLogger,
        iteration: int,
    ) -> Dict[str, Any]:
        content_chunks: List[str] = []
        tool_calls: List[Any] = []
        tool_call_ids: set = set()
        builtin_tool_calls: List[Dict[str, Any]] = []
        builtin_tool_call_ids: set = set()
        tool_items: Dict[str, Dict[str, Any]] = {}
        response_id: Optional[str] = None
        error: Optional[str] = None
        error_details: Optional[Dict[str, Any]] = None
        response: Optional[Any] = None

        with client.responses.stream(**request_payload) as stream:
            for event in stream:
                event_type = getattr(event, "type", "")
                if event_type == "response.created":
                    response_id = getattr(event.response, "id", None)
                elif event_type == "response.output_text.delta":
                    content_chunks.append(event.delta)
                elif event_type == "response.output_item.added":
                    item = event.item
                    if getattr(item, "type", None) == "function_call":
                        item_id = getattr(item, "id", None) or getattr(item, "call_id", None)
                        if item_id:
                            tool_items[item_id] = {
                                "call_id": getattr(item, "call_id", None),
                                "name": getattr(item, "name", None),
                                "arguments": getattr(item, "arguments", "") or "",
                            }
                    self._record_builtin_tool_call(
                        item, builtin_tool_calls, builtin_tool_call_ids
                    )
                elif event_type == "response.function_call_arguments.delta":
                    entry = tool_items.setdefault(
                        event.item_id, {"call_id": None, "name": None, "arguments": ""}
                    )
                    entry["arguments"] += event.delta
                elif event_type == "response.function_call_arguments.done":
                    entry = tool_items.get(event.item_id, {})
                    name = getattr(event, "name", None) or entry.get("name")
                    arguments = getattr(event, "arguments", None) or entry.get("arguments", "")
                    call_id = entry.get("call_id") or event.item_id
                    if call_id and call_id not in tool_call_ids:
                        tool_calls.append(
                            SimpleNamespace(
                                name=name or "", arguments=arguments or "", call_id=call_id
                            )
                        )
                        tool_call_ids.add(call_id)
                elif event_type == "response.output_item.done":
                    item = event.item
                    if getattr(item, "type", None) == "function_call":
                        call_id = getattr(item, "call_id", None) or getattr(item, "id", None)
                        if call_id and call_id not in tool_call_ids:
                            tool_calls.append(
                                SimpleNamespace(
                                    name=getattr(item, "name", "") or "",
                                    arguments=getattr(item, "arguments", "") or "",
                                    call_id=call_id,
                                )
                            )
                            tool_call_ids.add(call_id)
                    self._record_builtin_tool_call(
                        item, builtin_tool_calls, builtin_tool_call_ids
                    )
                elif event_type == "response.completed":
                    response = event.response
                    response_id = getattr(response, "id", response_id)
                elif event_type == "response.failed":
                    error = "response.failed"
                    response = event.response
                    response_id = getattr(response, "id", response_id)
                elif event_type == "response.incomplete":
                    error = "response.incomplete"
                    response = event.response
                    response_id = getattr(response, "id", response_id)
                    error_details = self._extract_incomplete_details(response) or self._extract_incomplete_details(event)
                elif event_type == "error":
                    error = event.message
                    break

            if not response and not error:
                response = stream.get_final_response()
                response_id = getattr(response, "id", response_id)

        content_text = "".join(content_chunks).strip()
        if not content_text and response is not None:
            content_text = self._extract_response_text(response)
        if not tool_calls and response is not None:
            tool_calls = self._extract_tool_calls(response)
        usage = self._extract_usage(response)
        if error == "response.incomplete" and not error_details:
            error_details = self._extract_incomplete_details(response)

        stream_event = {
            "type": "stream_summary",
            "iteration": iteration,
            "response_id": response_id,
            "content_preview": content_text[:2000],
            "tool_call_count": len(tool_calls),
            "usage": usage,
        }
        if builtin_tool_calls:
            stream_event["builtin_tool_calls"] = builtin_tool_calls
        if error_details:
            stream_event["incomplete_details"] = error_details
        event_logger.log_event(stream_event)

        return {
            "response_id": response_id,
            "content_text": content_text,
            "tool_calls": tool_calls,
            "builtin_tool_calls": builtin_tool_calls,
            "error": error,
            "usage": usage,
            "error_details": error_details,
        }

    def _extract_response_text(self, response: Any) -> str:
        output_text = getattr(response, "output_text", "") or ""
        if output_text:
            return output_text
        parts: List[str] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) != "message":
                continue
            for content in getattr(item, "content", []) or []:
                if getattr(content, "type", None) == "output_text":
                    parts.append(content.text)
        return "".join(parts).strip()

    def _extract_tool_calls(self, response: Any) -> List[Any]:
        tool_calls: List[Any] = []
        for item in getattr(response, "output", []) or []:
            if getattr(item, "type", None) == "function_call":
                tool_calls.append(item)
        return tool_calls

    def _get_item_attr(self, item: Any, key: str) -> Any:
        if hasattr(item, key):
            value = getattr(item, key)
            if value is not None:
                return value
        if isinstance(item, dict):
            return item.get(key)
        try:
            return item[key]
        except (KeyError, TypeError, IndexError):
            return None

    def _record_builtin_tool_call(
        self,
        item: Any,
        calls: List[Dict[str, Any]],
        seen: set,
    ) -> None:
        item_type = self._get_item_attr(item, "type")
        if not item_type or item_type == "function_call":
            return
        if not str(item_type).endswith("_call"):
            return
        payload: Dict[str, Any] = {"item_type": item_type}
        for key in ("id", "call_id", "name", "arguments", "query", "server_label", "tool_name"):
            value = self._get_item_attr(item, key)
            if value is None:
                continue
            if isinstance(value, str):
                value, _ = truncate(value, limit=800)
            payload[key] = value
        call_id = payload.get("call_id") or payload.get("id")
        if call_id:
            dedupe_key = f"{item_type}:{call_id}"
            if dedupe_key in seen:
                return
            seen.add(dedupe_key)
        else:
            signature = json.dumps(payload, sort_keys=True, ensure_ascii=True)
            if signature in seen:
                return
            seen.add(signature)
        calls.append(payload)

    def _extract_incomplete_details(self, obj: Any) -> Optional[Dict[str, Any]]:
        if obj is None:
            return None
        details = getattr(obj, "incomplete_details", None)
        if details is None and isinstance(obj, dict):
            details = obj.get("incomplete_details")
        if details is None:
            return None
        if hasattr(details, "model_dump"):
            details_dict = details.model_dump()
        elif hasattr(details, "dict"):
            details_dict = details.dict()
        elif isinstance(details, dict):
            details_dict = details
        elif isinstance(details, str):
            details_dict = {"reason": details}
        else:
            details_dict = {}
            for key in ("reason", "message"):
                value = getattr(details, key, None)
                if value:
                    details_dict[key] = value
            if not details_dict:
                try:
                    details_dict = dict(details)
                except Exception:
                    return None
        return details_dict or None

    def _extract_usage(self, response: Any) -> Dict[str, int]:
        if response is None:
            return {}
        usage = getattr(response, "usage", None)
        if usage is None:
            return {}
        if hasattr(usage, "model_dump"):
            usage_dict = usage.model_dump()
        elif hasattr(usage, "dict"):
            usage_dict = usage.dict()
        elif isinstance(usage, dict):
            usage_dict = usage
        else:
            try:
                usage_dict = dict(usage)
            except Exception:
                return {}

        def as_int(value: Any) -> Optional[int]:
            if isinstance(value, bool):
                return None
            if isinstance(value, (int, float)):
                return int(value)
            return None

        def pick_int(data: Dict[str, Any], *keys: str) -> Optional[int]:
            for key in keys:
                val = as_int(data.get(key))
                if val is not None:
                    return val
            return None

        normalized: Dict[str, int] = {}
        input_tokens = pick_int(usage_dict, "input_tokens", "prompt_tokens")
        output_tokens = pick_int(usage_dict, "output_tokens", "completion_tokens")
        total_tokens = pick_int(usage_dict, "total_tokens")
        if total_tokens is None and (input_tokens is not None or output_tokens is not None):
            total_tokens = (input_tokens or 0) + (output_tokens or 0)

        if input_tokens is not None:
            normalized["input_tokens"] = input_tokens
        if output_tokens is not None:
            normalized["output_tokens"] = output_tokens
        if total_tokens is not None:
            normalized["total_tokens"] = total_tokens

        input_details = usage_dict.get("input_tokens_details")
        if isinstance(input_details, dict):
            cached_tokens = pick_int(input_details, "cached_tokens", "cache_read_tokens")
            if cached_tokens is not None:
                normalized["cached_tokens"] = cached_tokens

        output_details = usage_dict.get("output_tokens_details")
        if isinstance(output_details, dict):
            reasoning_tokens = pick_int(output_details, "reasoning_tokens")
            if reasoning_tokens is not None:
                normalized["reasoning_tokens"] = reasoning_tokens

        return normalized

    def _build_system_prompt(
        self, instruction: str, max_iterations: int, active_skills: Optional[List[str]] = None
    ) -> str:
        active_skills = active_skills or []

        base_prompt = f"""You are a highly efficient, exacting autonomous coding agent. Complete the task as specified; be direct and structured, avoid unrequested work (modifications not explicitly permitted by task constraints are forbidden), and flag ambiguity or missing information before proceeding.
## Task
{instruction}

## Workflow
Before planning, extract key constraints only (max 6 bullets, 1 line each); do not quote or restate large blocks of the task. Then list required artifacts/paths briefly (short list).
1) First response: call update_plan with 3-5 short steps (<= 12 words each), mark the first in_progress; explanation <= 2 sentences; do not call other tools.
2) Execute the plan; do not call update_plan unless the plan changes. Start implementation by the second response; avoid extended planning.
3) If the plan changes, call update_plan with a brief explanation and revised steps, then continue.
4) The plan must include a Test/Verify step when possible. For API/service implementations, this should include running tests or making actual client calls to verify correctness, not just checking that the server starts.
## Tools
- web_search: find information on the web
- search_docs: find library IDs in docs database (e.g., search_docs("mujoco"))
- get_docs: get documentation for a library (e.g., get_docs("/google-deepmind/mujoco", "solver options"))
- run_command: execute shell commands
- apply_patch: edit files (V4A patch format)
- update_plan: update the plan
- task_complete: signal that the task is fully completed with a summary

## Tool & Build Heuristics
- When you see a tool directory with `run/`, `bin/`, or `build/` subdirectories, ALWAYS run `ls <tool>/run` to check for pre-built binaries BEFORE attempting to compile from source.
- Before running an unfamiliar CLI tool, run `<tool> --help` or use search_docs/get_docs to understand proper usage.

## Documentation lookup
Before code edits involving a third-party library/API, use search_docs to find the library, then get_docs to fetch relevant documentation.

## Editing
- Use apply_patch for file edits; avoid shell redirects or in-place edits (no sed -i, cat >).
- Write required outputs to exact absolute paths specified (e.g., /app/sol.sql); verify file exists.

## Verification
- Fix the exact error from traceback, then rerun; repeat until passing.
- Exit code 0 without correct output is failure; run tests or verify expected behavior.

## Constraints
- Max {max_iterations} steps (`HOOKELE_MAX_ITERATIONS`); warn yourself with 5 steps left and pivot sooner if stuck.
- Use failure output to change strategy if the same error repeats.
- Batch tool calls and avoid repeating identical commands.
- Use /tmp for scratch; do not create or modify /tests unless instructed.
- If a command or install fails (missing tool, PEP 668, timeouts), pivot to a lightweight alternative.
- If correctness across multiple inputs is required, do not hardcode examples; the solution must generalize."""

        # Inject skills
        if active_skills:
            skill_content = "\n".join(SKILLS[s]["content"] for s in active_skills if s in SKILLS)
            base_prompt += f"\n{skill_content}"

        return base_prompt

    def _build_trajectory(
        self,
        steps: List[Step],
        token_usage: Dict[str, int],
        cost_usd: Optional[float],
        model_name: Optional[str],
    ) -> Trajectory:
        agent = Agent(name=self.name(), version=self.version(), model_name=model_name)
        final_metrics: FinalMetrics | None = None
        if token_usage:
            final_metrics = FinalMetrics(
                total_prompt_tokens=token_usage.get("input_tokens"),
                total_completion_tokens=token_usage.get("output_tokens"),
                total_cached_tokens=token_usage.get("cached_tokens"),
                total_cost_usd=cost_usd,
                total_steps=len(steps),
            )
        return Trajectory(
            session_id=f"hookele-{uuid.uuid4()}",
            agent=agent,
            steps=steps,
            final_metrics=final_metrics,
        )

    def _compute_cost_usd(
        self, token_usage: Dict[str, int], model_name: Optional[str]
    ) -> Optional[float]:
        def read_rate(name: str) -> Optional[float]:
            raw = os.getenv(name)
            if raw is None:
                return None
            try:
                return float(raw)
            except ValueError:
                return None

        input_rate_1m = read_rate("HOOKELE_COST_PER_1M_INPUT_USD")
        output_rate_1m = read_rate("HOOKELE_COST_PER_1M_OUTPUT_USD")
        cache_rate_1m = read_rate("HOOKELE_COST_PER_1M_CACHED_INPUT_USD")
        input_rate_1k = read_rate("HOOKELE_COST_PER_1K_INPUT_USD")
        output_rate_1k = read_rate("HOOKELE_COST_PER_1K_OUTPUT_USD")
        cache_rate_1k = read_rate("HOOKELE_COST_PER_1K_CACHED_INPUT_USD")

        tokens_per_unit = None
        input_rate = output_rate = cache_rate = None
        if any(rate is not None for rate in (input_rate_1m, output_rate_1m, cache_rate_1m)):
            tokens_per_unit = 1_000_000.0
            input_rate, output_rate, cache_rate = input_rate_1m, output_rate_1m, cache_rate_1m
        elif any(rate is not None for rate in (input_rate_1k, output_rate_1k, cache_rate_1k)):
            tokens_per_unit = 1_000.0
            input_rate, output_rate, cache_rate = input_rate_1k, output_rate_1k, cache_rate_1k
        else:
            model_rates = self._lookup_default_rates(model_name)
            if model_rates:
                input_rate, output_rate, cache_rate = model_rates
                tokens_per_unit = 1_000_000.0

        if tokens_per_unit is None:
            return None

        input_tokens = token_usage.get("input_tokens", 0)
        output_tokens = token_usage.get("output_tokens", 0)
        cached_tokens = token_usage.get("cached_tokens", 0)
        input_billable = input_tokens
        cost = 0.0

        if cache_rate is not None:
            input_billable = max(input_tokens - cached_tokens, 0)
            cost += (cached_tokens / tokens_per_unit) * cache_rate

        if input_rate is not None:
            cost += (input_billable / tokens_per_unit) * input_rate
        if output_rate is not None:
            cost += (output_tokens / tokens_per_unit) * output_rate

        return round(cost, 6)

    def _lookup_default_rates(
        self, model_name: Optional[str]
    ) -> Optional[tuple[Optional[float], Optional[float], Optional[float]]]:
        if not model_name:
            return None
        normalized = model_name.split("/")[-1].strip().lower()
        default_rates = {
            "gpt-5.1-codex-mini": (0.25, 2.00, 0.025),
            "gpt-5.1-codex-max": (1.25, 10.00, 0.125),
        }
        return default_rates.get(normalized)

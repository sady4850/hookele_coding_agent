"""Tool definitions and implementations for the Hookele agent."""

import json
import os
import re
import shlex
import time
import urllib.request
import urllib.parse
import urllib.error
from typing import Any, Dict, List, Optional

from harbor.environments.base import BaseEnvironment

from .apply_diff import apply_diff
from .utils import strip_bash_noise, truncate


# Tool definitions for OpenAI Responses API (Standard OpenAI schema)
CONTEXT7_API_KEY = os.getenv("CONTEXT7_API_KEY", "")
CONTEXT7_BASE_URL = "https://context7.com/api/v2"

TOOLS = [
    {
        "type": "web_search",  # OpenAI built-in web search
    },
    {
        "type": "function",
        "name": "search_docs",
        "description": "Search for a library in the documentation database. Returns library IDs that can be used with get_docs.",
        "parameters": {
            "type": "object",
            "properties": {
                "library_name": {
                    "type": "string",
                    "description": "Library name to search for (e.g., 'next.js', 'react', 'mujoco')"
                },
                "query": {
                    "type": "string",
                    "description": "Optional query to rank results by relevance (e.g., 'setup ssr', 'solver options')"
                }
            },
            "required": ["library_name"]
        },
    },
    {
        "type": "function",
        "name": "get_docs",
        "description": "Get documentation snippets for a library. Use search_docs first to find the library ID.",
        "parameters": {
            "type": "object",
            "properties": {
                "library_id": {
                    "type": "string",
                    "description": "Library ID from search_docs (e.g., '/vercel/next.js', '/google-deepmind/mujoco')"
                },
                "query": {
                    "type": "string",
                    "description": "Question or topic to get documentation for (e.g., 'how to configure solver', 'performance optimization')"
                }
            },
            "required": ["library_id", "query"]
        },
    },
    {
        "type": "function",
        "name": "apply_patch",
        "description": "Apply file changes using V4A patch format. Supports creating, updating, and deleting files.",
        "parameters": {
            "type": "object",
            "properties": {
                "patch": {
                    "type": "string",
                    "description": "V4A patch format. Example:\n*** Add File: new.py\n+content\n*** Update File: src/main.py\n@@ def foo():\n-old\n+new\n*** Delete File: obsolete.py\n*** End Patch"
                }
            },
            "required": ["patch"]
        },
    },
    {
        "type": "function",
        "name": "run_command",
        "description": "Run a shell command and return its output.",
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"type": "string", "description": "The shell command to execute."}
            },
            "required": ["command"],
        },
    },
    {
        "type": "function",
        "name": "update_plan",
        "description": "Updates the task plan. Provide an optional explanation and a list of plan items, each with a step and status. At most one step can be in_progress at a time.",
        "parameters": {
            "type": "object",
            "properties": {
                "explanation": {
                    "type": "string",
                    "description": "Optional explanation for why the plan changed."
                },
                "plan": {
                    "type": "array",
                    "description": "The list of plan steps.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "step": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["pending", "in_progress", "completed"]
                            }
                        },
                        "required": ["step", "status"]
                    }
                }
            },
            "required": ["plan"]
        },
    },
    {
        "type": "function",
        "name": "task_complete",
        "description": "Signal that the task is fully completed. Provide a brief summary of what was achieved and any key outputs.",
        "parameters": {
            "type": "object",
            "properties": {
                "summary": {
                    "type": "string",
                    "description": "Brief summary of the completed task and verification results."
                }
            },
            "required": ["summary"]
        },
    },
]


async def run_command(
    environment: BaseEnvironment, command: str, timeout_sec: int = 120
) -> Dict[str, Any]:
    """Run a shell command."""
    try:
        result = await environment.exec(
            command=command, timeout_sec=timeout_sec
        )
        stdout = strip_bash_noise(result.stdout)
        stderr = strip_bash_noise(result.stderr)
        return {
            "stdout": stdout,
            "stderr": stderr,
            "return_code": result.return_code,
        }
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "return_code": 1}


def _context7_request(endpoint: str, params: Dict[str, str]) -> Dict[str, Any]:
    """Make a request to Context7 API."""
    query_string = urllib.parse.urlencode(params)
    url = f"{CONTEXT7_BASE_URL}/{endpoint}?{query_string}"

    headers = {"User-Agent": "Hookele/1.0"}
    if CONTEXT7_API_KEY:
        headers["Authorization"] = f"Bearer {CONTEXT7_API_KEY}"

    req = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        return {"error": f"HTTP {e.code}: {e.reason}"}
    except urllib.error.URLError as e:
        return {"error": f"URL error: {e.reason}"}
    except Exception as e:
        return {"error": str(e)}


def search_docs(library_name: str, query: Optional[str] = None) -> str:
    """Search for libraries in the documentation database."""
    params = {"libraryName": library_name}
    if query:
        params["query"] = query

    result = _context7_request("libs/search", params)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    # Format results for the model
    libraries = result.get("results", [])
    if not libraries:
        return json.dumps({"message": f"No libraries found for '{library_name}'"})

    # Return top 5 results with key info
    formatted = []
    for lib in libraries[:5]:
        formatted.append({
            "id": lib.get("id"),
            "title": lib.get("title"),
            "description": lib.get("description", "")[:200],
            "tokens": lib.get("totalTokens"),
        })

    return json.dumps({"libraries": formatted}, indent=2)


def get_docs(library_id: str, query: str) -> str:
    """Get documentation for a specific library."""
    params = {
        "libraryId": library_id,
        "query": query,
        "type": "json",  # Structured data allows better formatting
    }

    result = _context7_request("context", params)

    if "error" in result:
        return json.dumps({"error": result["error"]})

    code_snippets = result.get("codeSnippets", [])
    info_snippets = result.get("infoSnippets", [])

    if not code_snippets and not info_snippets:
        return json.dumps({"message": f"No documentation found for query '{query}'"})

    output_parts = []

    # Format info snippets first (context/explanations)
    for snippet in info_snippets[:5]:  # Limit to 5 info snippets
        breadcrumb = snippet.get("breadcrumb", "")
        content = snippet.get("content", "")
        if breadcrumb:
            output_parts.append(f"### {breadcrumb}\n{content}")
        else:
            output_parts.append(content)

    # Format code snippets
    for snippet in code_snippets[:10]:  # Limit to 10 code snippets
        title = snippet.get("codeTitle", "")
        desc = snippet.get("codeDescription", "")
        code_list = snippet.get("codeList", [])

        output_parts.append(f"## {title}\n{desc}")
        for code_block in code_list:
            lang = code_block.get("language", "")
            code = code_block.get("code", "")
            output_parts.append(f"```{lang}\n{code}\n```")

    output = "\n\n".join(output_parts)
    max_chars = _get_max_tool_output_chars()
    if len(output) > max_chars:
        output = output[:max_chars] + "\n... [truncated]"

    return output


async def execute_function(
    environment: BaseEnvironment, name: str, args: Dict[str, Any]
) -> Dict[str, Any]:
    """Execute a function tool and return result."""
    if name == "run_command":
        command = args.get("command", "")
        start = time.monotonic()
        result = await run_command(environment, command)
        duration_seconds = round(time.monotonic() - start, 1)
        ok = result["return_code"] == 0
        summary = None
        if not ok:
            summary = _summarize_command_failure(
                result["return_code"], result["stdout"], result["stderr"]
            )
        output = _format_exec_output_for_model(
            result["stdout"],
            result["stderr"],
            result["return_code"],
            duration_seconds,
        )
        return {"ok": ok, "output": output, "summary": summary}

    elif name == "apply_patch":
        result = await apply_patch_operation(environment, args)
        status = "completed" if result["success"] else "failed"
        output = f"Patch Output: {status} - {result.get('message', '')}"
        summary = output if not result["success"] else None
        return {"ok": result["success"], "output": output, "summary": summary}

    elif name == "update_plan":
        if not isinstance(args.get("plan"), list):
            return {
                "ok": False,
                "output": "Error: update_plan requires a plan list",
                "summary": "update_plan error: plan list missing",
            }
        return {"ok": True, "output": "Plan updated", "summary": None}

    elif name == "task_complete":
        summary = args.get("summary", "")
        return {"ok": True, "output": f"Task marked as complete. Summary: {summary}", "summary": None}

    elif name == "search_docs":
        library_name = args.get("library_name", "")
        query = args.get("query")
        if not library_name:
            return {
                "ok": False,
                "output": "Error: library_name is required",
                "summary": "search_docs error: library_name missing",
            }
        output = search_docs(library_name, query)
        ok = "error" not in output.lower()
        return {"ok": ok, "output": output, "summary": None if ok else output}

    elif name == "get_docs":
        library_id = args.get("library_id", "")
        query = args.get("query", "")
        if not library_id or not query:
            return {
                "ok": False,
                "output": "Error: library_id and query are required",
                "summary": "get_docs error: missing parameters",
            }
        output = get_docs(library_id, query)
        ok = "error" not in output.lower()[:100]  # Only check start for error
        return {"ok": ok, "output": output, "summary": None if ok else output}

    summary = f"Unknown function: {name}"
    return {"ok": False, "output": summary, "summary": summary}


def _format_exec_output_for_model(
    stdout: str,
    stderr: str,
    return_code: int,
    duration_seconds: float,
) -> str:
    aggregated = _aggregate_output(stdout, stderr)
    max_chars = _get_max_tool_output_chars()
    truncated_output, was_truncated = truncate(aggregated, limit=max_chars)
    payload = {
        "output": truncated_output,
        "metadata": {
            "exit_code": return_code,
            "duration_seconds": duration_seconds,
            "truncated": was_truncated,
        },
    }
    if was_truncated:
        payload["metadata"]["original_length"] = len(aggregated)
        payload["metadata"]["max_chars"] = max_chars
    return json.dumps(payload, ensure_ascii=True, separators=(",", ":"))


def _tail_lines(text: str, max_lines: int) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _summarize_command_failure(
    return_code: int, stdout: str, stderr: str, max_lines: int = 12
) -> str:
    parts = [f"Command failed (exit code: {return_code})."]
    stderr_tail = _tail_lines(stderr, max_lines)
    stdout_tail = _tail_lines(stdout, max_lines)
    if stderr_tail:
        parts.append("stderr (tail):\n" + stderr_tail)
    if stdout_tail:
        parts.append("stdout (tail):\n" + stdout_tail)
    if not stderr_tail and not stdout_tail:
        parts.append("No output captured.")
    return "\n".join(parts)


def _aggregate_output(stdout: str, stderr: str) -> str:
    if stdout and stderr:
        if stdout.endswith("\n"):
            return stdout + stderr
        return stdout + "\n" + stderr
    return stdout or stderr or ""


def _get_max_tool_output_chars() -> int:
    raw = os.getenv("HOOKELE_MAX_TOOL_OUTPUT_CHARS", "20000")
    try:
        value = int(raw)
    except ValueError:
        return 20000
    if value < 1000:
        return 1000
    return value


async def write_file(
    environment: BaseEnvironment, path: str, content: str
) -> Dict[str, Any]:
    """Write content to file."""
    # Use printf to preserve exact content (avoid implicit trailing newline).
    cmd = f"printf '%s' {shlex.quote(content)} > {shlex.quote(path)}"
    result = await run_command(environment, cmd, timeout_sec=30)  # Shorter timeout for writes
    
    if result["return_code"] == 0:
        return {"success": True, "message": f"Written to {path}"}
    
    # Build detailed error message
    error_parts = []
    if result["stderr"]:
        error_parts.append(f"stderr: {result['stderr']}")
    if result["stdout"]:
        error_parts.append(f"stdout: {result['stdout']}")
    error_parts.append(f"exit_code: {result['return_code']}")
    
    error_msg = "; ".join(error_parts) if error_parts else "Write failed (no output)"
    return {"success": False, "message": error_msg}


def strip_diff_prefixes(content: str) -> str:
    """Strip unified diff prefixes (+/-/ ) from content if present.

    When LLMs generate create_file operations, they sometimes format the content
    as a diff with '+' prefixes on every line. This strips those prefixes.
    """
    if not content:
        return content

    lines = content.split("\n")

    # Check if this looks like a diff (most lines start with +)
    plus_lines = sum(1 for line in lines if line.startswith("+"))
    if plus_lines < len(lines) * 0.8:  # Less than 80% lines start with +
        return content

    # Strip the + prefix from lines
    stripped = []
    for line in lines:
        if line.startswith("+"):
            stripped.append(line[1:])
        elif line.startswith("-"):
            # Skip lines marked for deletion (shouldn't exist in create_file)
            continue
        else:
            stripped.append(line)

    return "\n".join(stripped)


def _get_attr(obj: Any, key: str) -> Any:
    """Get attribute from object or dict."""
    # Try attribute access first
    if hasattr(obj, key):
        val = getattr(obj, key)
        if val is not None:
            return val
    # Try dict access
    if isinstance(obj, dict):
        return obj.get(key)
    # Try subscript access (for Pydantic models, etc.)
    try:
        return obj[key]
    except (KeyError, TypeError, IndexError):
        pass
    return None


def parse_v4a_patch(patch: str) -> List[Dict[str, Any]]:
    """Parse V4A patch format into a list of file operations."""
    operations = []
    lines = patch.split("\n")
    i = 0

    # Skip "*** Begin Patch" if present
    while i < len(lines) and (not lines[i].strip() or lines[i].strip() == "*** Begin Patch"):
        i += 1

    while i < len(lines):
        line = lines[i]

        # End of patch
        if line.strip() == "*** End Patch" or line.strip() == "":
            i += 1
            continue

        # Add File
        match = re.match(r'\*\*\*\s+Add\s+File:\s*(.+)', line)
        if match:
            path = match.group(1).strip()
            i += 1
            diff_lines = []
            while i < len(lines) and not lines[i].startswith("***"):
                diff_lines.append(lines[i])
                i += 1
            operations.append({
                "type": "create_file",
                "path": path,
                "diff": "\n".join(diff_lines)
            })
            continue

        # Update File
        match = re.match(r'\*\*\*\s+Update\s+File:\s*(.+)', line)
        if match:
            path = match.group(1).strip()
            i += 1
            diff_lines = []
            while i < len(lines) and not lines[i].startswith("***"):
                diff_lines.append(lines[i])
                i += 1
            operations.append({
                "type": "update_file",
                "path": path,
                "diff": "\n".join(diff_lines)
            })
            continue

        # Delete File
        match = re.match(r'\*\*\*\s+Delete\s+File:\s*(.+)', line)
        if match:
            path = match.group(1).strip()
            operations.append({
                "type": "delete_file",
                "path": path
            })
            i += 1
            continue

        # Unknown line, skip
        i += 1

    return operations


async def apply_patch_operation(
    environment: BaseEnvironment, operation: Any
) -> Dict[str, Any]:
    """Apply a patch operation from apply_patch tool."""
    patch = _get_attr(operation, "patch")

    if not patch:
        return {"success": False, "message": "No patch provided"}

    # Parse the V4A patch format
    try:
        ops = parse_v4a_patch(patch)
    except Exception as exc:
        return {"success": False, "message": f"Failed to parse patch: {exc}"}

    if not ops:
        return {"success": False, "message": "No file operations found in patch"}

    results = []
    result_entries = []
    for op in ops:
        op_type = op.get("type")
        path = op.get("path")
        diff = op.get("diff", "")

        if op_type == "create_file":
            try:
                content = apply_diff("", diff, mode="create")
            except Exception as exc:
                results.append(f"FAIL {path}: {exc}")
                result_entries.append({
                    "path": path,
                    "action": op_type,
                    "ok": False,
                    "message": str(exc),
                })
                continue
            res = await write_file(environment, path, content)
            results.append(f"{'OK' if res['success'] else 'FAIL'} {path}: {res['message']}")
            result_entries.append({
                "path": path,
                "action": op_type,
                "ok": res["success"],
                "message": res["message"],
            })

        elif op_type == "update_file":
            current = await run_command(environment, f"cat {shlex.quote(path)}")
            if current["return_code"] != 0:
                results.append(f"FAIL {path}: Cannot read file")
                result_entries.append({
                    "path": path,
                    "action": op_type,
                    "ok": False,
                    "message": "Cannot read file",
                })
                continue
            try:
                patched = apply_diff(current["stdout"] or "", diff)
            except Exception as exc:
                results.append(f"FAIL {path}: {exc}")
                result_entries.append({
                    "path": path,
                    "action": op_type,
                    "ok": False,
                    "message": str(exc),
                })
                continue
            res = await write_file(environment, path, patched)
            results.append(f"{'OK' if res['success'] else 'FAIL'} {path}: {res['message']}")
            result_entries.append({
                "path": path,
                "action": op_type,
                "ok": res["success"],
                "message": res["message"],
            })

        elif op_type == "delete_file":
            res = await run_command(environment, f"rm -f {shlex.quote(path)}")
            ok = res["return_code"] == 0
            results.append(f"{'OK' if ok else 'FAIL'} {path}: deleted")
            result_entries.append({
                "path": path,
                "action": op_type,
                "ok": ok,
                "message": "deleted",
            })

    all_ok = all(r.startswith("OK") for r in results)
    return {
        "success": all_ok,
        "message": "\n".join(results),
        "results": result_entries,
    }

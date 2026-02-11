"""Utility functions for the Hookele agent."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def iso_now() -> str:
    """Return current UTC time in ISO format."""
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def truncate(text: Optional[str], limit: int = 50_000) -> tuple[Optional[str], bool]:
    """Truncate text to limit, keeping head and tail."""
    if text is None:
        return None, False
    if len(text) <= limit:
        return text, False
    head = text[: limit // 2]
    tail = text[-limit // 2 :]
    return head + "\n...[TRUNCATED]...\n" + tail, True


_BASH_NOISE_LINES = [
    "bash: cannot set terminal process group (-1): Inappropriate ioctl for device",
    "bash: no job control in this shell",
]


def strip_bash_noise(text: Optional[str]) -> Optional[str]:
    """Remove bash startup messages from command output."""
    if text is None:
        return None
    lines = text.split("\n")
    filtered = [line for line in lines if line not in _BASH_NOISE_LINES]
    return "\n".join(filtered)


class EventLogger:
    """JSONL trajectory logger."""

    def __init__(self, base: Path):
        self.base = base
        self.base.mkdir(parents=True, exist_ok=True)
        self.traj_path = self.base / "traj.jsonl"
        self.summary_path = self.base / "summary.json"

    def log_event(self, event: Dict[str, Any]) -> None:
        event.setdefault("ts", iso_now())
        with self.traj_path.open("a") as f:
            f.write(json.dumps(event))
            f.write("\n")

    def write_summary(self, summary: Dict[str, Any]) -> None:
        self.summary_path.write_text(json.dumps(summary, indent=2))


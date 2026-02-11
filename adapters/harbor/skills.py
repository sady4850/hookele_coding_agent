"""Skills registry for Hookele agent.

Each skill is a prompt block that is dynamically injected into the system prompt
based on the task requirements.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple


def _parse_frontmatter(text: str) -> Tuple[Dict[str, str], str]:
    if not text.startswith("---"):
        return {}, text
    parts = text.split("\n")
    meta: Dict[str, str] = {}
    idx = 1
    while idx < len(parts):
        line = parts[idx].strip()
        if line == "---":
            idx += 1
            break
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip()] = value.strip()
        idx += 1
    content = "\n".join(parts[idx:]).lstrip("\n")
    return meta, content


def _load_skills() -> Dict[str, Dict[str, str]]:
    skills: Dict[str, Dict[str, str]] = {}
    skills_dir = Path(__file__).resolve().parents[2] / "skills"
    if not skills_dir.exists():
        return skills
    for path in sorted(skills_dir.glob("*.md")):
        text = path.read_text(encoding="utf-8")
        meta, content = _parse_frontmatter(text)
        key = path.stem
        name = meta.get("name", key)
        description = meta.get("description", "")
        skills[key] = {
            "name": name,
            "description": description,
            "content": content,
        }
    return skills


SKILLS = _load_skills()

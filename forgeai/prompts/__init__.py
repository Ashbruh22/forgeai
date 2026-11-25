"""Versioned prompt templates.

Every prompt is a Markdown file under ``forgeai/prompts/``. No prompt text lives in
Python. A file looks like::

    ---
    version: 3
    ---
    # system
    You are ...
    # user
    Question: {{question}}

``{{name}}`` placeholders are substituted; a missing variable is an error.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

PROMPTS_DIR = Path(__file__).parent

_FRONTMATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_PLACEHOLDER = re.compile(r"\{\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}\}")
_SECTION = re.compile(r"^# (system|user)\s*$", re.MULTILINE)


class PromptError(Exception):
    """Raised for a missing, malformed or under-specified prompt."""


@dataclass(frozen=True)
class PromptTemplate:
    prompt_id: str
    version: int
    system: str
    user: str

    def render(self, variables: dict[str, Any]) -> tuple[str, str]:
        """Return ``(system, user)`` with placeholders filled in."""
        return _fill(self.system, variables, self.prompt_id), _fill(
            self.user, variables, self.prompt_id
        )


def _fill(text: str, variables: dict[str, Any], prompt_id: str) -> str:
    missing = sorted({m for m in _PLACEHOLDER.findall(text) if m not in variables})
    if missing:
        raise PromptError(f"Prompt {prompt_id!r} is missing variables: {missing}")
    return _PLACEHOLDER.sub(lambda m: str(variables[m.group(1)]), text)


@lru_cache(maxsize=None)
def load_prompt(prompt_id: str, prompts_dir: Path = PROMPTS_DIR) -> PromptTemplate:
    """Load ``<prompts_dir>/<prompt_id>.md`` (e.g. ``ingest/summary``)."""
    path = prompts_dir / f"{prompt_id}.md"
    if not path.is_file():
        raise PromptError(f"Prompt file not found: {path}")
    raw = path.read_text(encoding="utf-8").replace("\r\n", "\n")
    front = _FRONTMATTER.match(raw)
    if not front:
        raise PromptError(f"Prompt {prompt_id!r} has no '---' frontmatter with a version")
    version_match = re.search(r"^version:\s*(\d+)\s*$", front.group(1), re.MULTILINE)
    if not version_match:
        raise PromptError(f"Prompt {prompt_id!r} frontmatter lacks 'version: <int>'")
    body = raw[front.end() :]
    parts = _SECTION.split(body)
    # parts = [preamble, name1, text1, name2, text2, ...]
    sections = {parts[i]: parts[i + 1].strip() for i in range(1, len(parts) - 1, 2)}
    if not sections:
        raise PromptError(f"Prompt {prompt_id!r} needs a '# system' or '# user' section")
    return PromptTemplate(
        prompt_id=prompt_id,
        version=int(version_match.group(1)),
        system=sections.get("system", ""),
        user=sections.get("user", ""),
    )

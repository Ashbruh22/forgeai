"""Guards for the project rules in CLAUDE.md that can be checked mechanically."""

import ast
import re
from pathlib import Path

PACKAGE = Path(__file__).parents[1] / "forgeai"
DOMAIN_WORDS = re.compile(
    r"\b(payroll|salary|seafarer|procurement|spares?|supplier|leave policy|annual leave|shore staff)\b",
    re.IGNORECASE,
)


def python_files() -> list[Path]:
    return [p for p in PACKAGE.rglob("*.py")]


def test_no_demo_specific_logic_in_the_factory() -> None:
    """Rule 1: demo specifics live only in data/, never in forgeai/."""
    offenders = [
        f"{p.relative_to(PACKAGE)}: {m.group(0)}"
        for p in list(python_files()) + list(PACKAGE.rglob("*.md"))
        for m in DOMAIN_WORDS.finditer(p.read_text(encoding="utf-8"))
    ]
    assert offenders == []


def test_no_prompt_text_hardcoded_in_python() -> None:
    """Rule 3: prompts are versioned .md files; Python holds no instruction strings."""
    offenders: list[str] = []
    for path in python_files():
        tree = ast.parse(path.read_text(encoding="utf-8"))
        docstrings = {
            id(node.body[0].value)
            for node in ast.walk(tree)
            if isinstance(node, (ast.Module, ast.ClassDef, ast.FunctionDef))
            and node.body and isinstance(node.body[0], ast.Expr)
            and isinstance(getattr(node.body[0], "value", None), ast.Constant)
        }
        for node in ast.walk(tree):
            if isinstance(node, ast.Constant) and isinstance(node.value, str) and id(node) not in docstrings:
                if re.search(r"\b(You are|Your task|Respond with|Reply with)\b", node.value):
                    offenders.append(f"{path.relative_to(PACKAGE)}:{node.lineno}")
    assert offenders == []


def test_every_prompt_file_declares_a_version() -> None:
    for path in (PACKAGE / "prompts").rglob("*.md"):
        assert re.match(r"\A---\s*\nversion:\s*\d+\s*\n---", path.read_text(encoding="utf-8")), path

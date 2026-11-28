"""The test split is sealed: the optimizer's code path cannot read test results (spec §6.1, §6.6)."""

from __future__ import annotations

import ast
import json
from pathlib import Path

import pytest

from forgeai.judge import sealed, store
from forgeai.judge.models import CaseRun
from forgeai.runner.contract import Answer

PACKAGE = Path(__file__).parents[1] / "forgeai"
# What the optimizer (P7) may import from the judge: the dev side of the store, the scoring code and the metrics.
OPTIMIZER_VISIBLE = ["forgeai.judge.store", "forgeai.judge.score", "forgeai.judge.metrics", "forgeai.judge.models",
                     "forgeai.judge.layer1", "forgeai.judge.layer2"]
FORBIDDEN = {"forgeai.judge.sealed", "forgeai.judge.baseline"}


def module_path(name: str) -> Path | None:
    path = PACKAGE.parent.joinpath(*name.split("."))
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    return None


def imports_of(path: Path) -> set[str]:
    found: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("forgeai"):
            found.add(node.module)
            found.update(f"{node.module}.{alias.name}" for alias in node.names)
        elif isinstance(node, ast.Import):
            found.update(a.name for a in node.names if a.name.startswith("forgeai"))
    return found


def reachable(start: list[str]) -> set[str]:
    seen: set[str] = set()
    todo = list(start)
    while todo:
        name = todo.pop()
        path = module_path(name)
        if name in seen or path is None:
            continue
        seen.add(name)
        todo.extend(imports_of(path))
    return seen


def test_nothing_the_optimizer_may_import_can_reach_the_sealed_module() -> None:
    assert not reachable(OPTIMIZER_VISIBLE) & FORBIDDEN


def test_the_optimizer_package_when_it_exists_cannot_reach_the_sealed_module_or_name_test_results() -> None:
    package = PACKAGE / "optimizer"
    if not package.is_dir():
        pytest.skip("the optimizer arrives in P7; this test starts guarding it the day it exists")
    names = [".".join(p.relative_to(PACKAGE.parent).with_suffix("").parts).removesuffix(".__init__") for p in package.rglob("*.py")]
    assert not reachable(names) & FORBIDDEN
    for path in package.rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        assert "sealed" not in text and "test_scorecard" not in text and "test_case_scores" not in text, path


def test_only_the_baseline_the_driver_and_the_command_line_import_the_sealed_module() -> None:
    importers = {path.relative_to(PACKAGE).as_posix() for path in PACKAGE.rglob("*.py")
                 if "forgeai.judge.sealed" in imports_of(path)}
    assert importers <= {"judge/baseline.py", "cli.py", "iterate.py"}, importers


def sample(case_id: str) -> CaseRun:
    return CaseRun(case_id=case_id, answer=Answer(status="refused", answer="no"))


def test_test_results_are_written_apart_from_dev_results_and_the_dev_readers_refuse_the_sealed_folder(tmp_path) -> None:
    store.write_split(store.dev_dir(tmp_path), "", [sample("d1")], [], {"split": "dev"})
    sealed.write_test(tmp_path, [sample("t1")], [], {"split": "test", "pass_rate": {"passes": 3, "n": 4}})
    dev_files = {p.name for p in store.dev_dir(tmp_path).iterdir()}
    test_files = {p.name for p in sealed.sealed_dir(tmp_path).iterdir()}
    assert "scorecard.json" in dev_files and "test_scorecard.json" in test_files and not dev_files & test_files
    assert store.read_dev_scorecard(tmp_path)["split"] == "dev"
    assert sealed.read_test_scorecard(tmp_path)["pass_rate"]["passes"] == 3
    assert "t1" not in "".join(p.read_text(encoding="utf-8") for p in store.dev_dir(tmp_path).iterdir())
    with pytest.raises(PermissionError):
        store.read_dev_scorecard(sealed.sealed_dir(tmp_path).parent / "sealed")  # a path inside sealed/ is refused
    with pytest.raises(PermissionError):
        store.read_dev_scores(sealed.sealed_dir(tmp_path))


def test_the_dev_loader_drops_test_cases_before_anyone_else_sees_them(tmp_path) -> None:
    (tmp_path / "03_evals").mkdir()
    rows = [{"id": f"c{n}", "type": "out_of_scope", "split": split, "question": "q", "expected_status": "refused"}
            for n, split in enumerate(["dev", "test", "dev"])]
    (tmp_path / "03_evals" / "eval_set.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    assert [c.id for c in store.load_dev_cases(tmp_path)] == ["c0", "c2"]
    with pytest.raises(PermissionError):
        store.load_dev_cases(tmp_path / "sealed")

"""Command line interface: ``forgeai chat`` works now; other commands arrive with their phase."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgeai.config import ConfigError, load_config
from forgeai.embeddings import SentenceTransformerEmbedder
from forgeai.llm import LLMClient, LLMUnavailableError, set_client
from forgeai.runner.agent_def import load_agent
from forgeai.runlock import RunLockedError, run_lock as _run_lock
from forgeai.runner.runner import AgentRunner, History, RunResult
from forgeai.timing import stage_timer

app = typer.Typer(add_completion=False, help="ForgeAI: the agent that builds agents.")
console = Console()

@contextmanager
def run_lock(run: Path, stage: str) -> Iterator[None]:
    """The run-folder lock; a second build of the same folder stops with a clear message instead of a traceback."""
    try:
        with _run_lock(run, stage):
            yield
    except RunLockedError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(4) from exc


def preflight(config, run: Path) -> None:  # noqa: ANN001
    """Load each model with the context it will use, report the GPU/CPU split and the Ollama settings, save both."""
    import httpx

    from forgeai.resources import load_and_check, system_memory, verify_ollama_env

    report: dict = {"ollama_env": verify_ollama_env(config.ollama_env) if config.ollama_env else {}, "models": []}
    try:
        for spec in (config.agent_model, config.judge_model):
            report["models"].append(load_and_check(config.ollama_host, spec.name, config.num_ctx_for(spec.name)))
    except httpx.HTTPError as exc:
        console.print(f"[red]Ollama preflight failed: {exc!r}[/red]")
        raise typer.Exit(5) from exc
    total, free = system_memory()
    report["ram_gb"] = {"total": round(total, 1), "free_at_start": round(free, 1)}
    run.mkdir(parents=True, exist_ok=True)
    (run / "resources.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    for m in report["models"]:
        style = "green" if m.get("fully_on_gpu") else "bold red"
        console.print(f"[{style}]{m['name']}: {m.get('gpu_pct')}% on GPU, context {m.get('context')}, {m.get('size_gb')} GB[/{style}]")
        if not m.get("fully_on_gpu"):
            console.print(f"[red]  {m['name']} is NOT fully on the GPU, so part of it runs on the CPU. Lower num_ctx for this role "
                          "in forgeai.yaml, or use a smaller quantization, before relying on this run.[/red]")
    for key, took in report["ollama_env"].get("took_effect", {}).items():
        console.print(f"[{'green' if took else 'red'}]ollama {key}: wanted {config.ollama_env[key]}, "
                      f"server has {report['ollama_env']['server_log'].get(key)}[/{'green' if took else 'red'}]")
    if free / total * 100 < config.min_free_ram_pct:
        console.print(f"[red]Only {free:.1f} of {total:.1f} GB RAM is free before the build starts.[/red]")


COMMANDS_HELP = "/sources  show retrieved chunks   /debug  raw JSON + tool trace   /reset  clear history   /quit"


def _render(result: RunResult) -> None:
    answer = result.answer
    style = {"answered": "green", "refused": "yellow", "needs_clarification": "cyan"}[answer.status]
    body = answer.answer or answer.clarifying_question or "(empty)"
    if answer.status == "needs_clarification" and answer.clarifying_question:
        body = answer.clarifying_question
    console.print(Panel(body, title=answer.status, border_style=style))
    for call in answer.tool_calls:
        console.print(f"[dim]tool {call.tool}({call.args}) -> {call.result}[/dim]")
    for citation in answer.citations:
        console.print(f"[dim]source: {citation.doc} ({citation.chunk_id}) \"{citation.quote}\"[/dim]")
    if result.issues:
        console.print(f"[red]contract issues: {result.issues}[/red]")


def _show_sources(result: RunResult) -> None:
    table = Table("chunk_id", "dense", "bm25", "text")
    for hit in result.retrieved:
        table.add_row(hit.chunk.chunk_id, f"{hit.dense:.3f}", f"{hit.bm25:.2f}", hit.chunk.text[:120])
    console.print(table if result.retrieved else "[yellow]nothing retrieved above min_score[/yellow]")


@app.command()
def chat(
    agent_path: Path = typer.Argument(..., help="Path to an agent.yaml produced by ForgeAI"),
    config_path: Path = typer.Option(Path("forgeai.yaml"), "--config", help="forgeai.yaml"),
) -> None:
    """Chat with any generated agent. Knows nothing about its domain."""
    try:
        config = load_config(config_path)
    except ConfigError as exc:
        console.print(f"[red]Config error:[/red] {exc}")
        raise typer.Exit(2) from exc
    agent = load_agent(agent_path)
    set_client(LLMClient(config))
    with console.status("Building retrieval index..."):
        runner = AgentRunner(
            agent, agent_path.parent, config, SentenceTransformerEmbedder(config.embedding_model, config.embedding_revision)
        )
    for name, reason in runner.skipped_files.items():
        console.print(f"[yellow]skipped {name}: {reason}[/yellow]")
    console.print(f"[bold]{agent.name}[/bold]  ({len(runner.index.chunks)} chunks, tools: {agent.tools or 'none'})")
    console.print(f"[dim]{COMMANDS_HELP}[/dim]")

    history: History = []
    last: RunResult | None = None
    while True:
        try:
            question = console.input("[bold blue]you> [/bold blue]").strip()
        except (EOFError, KeyboardInterrupt):
            break
        if not question:
            continue
        if question == "/quit":
            break
        if question == "/reset":
            history.clear()
            console.print("[dim]history cleared[/dim]")
        elif question == "/sources":
            _show_sources(last) if last else console.print("[dim]ask something first[/dim]")
        elif question == "/debug":
            if last:
                console.print_json(json.dumps({
                    "search_query": last.search_query, "steps": last.steps,
                    "answer": last.answer.model_dump(mode="json"), "issues": last.issues,
                    "latency_ms": round(last.latency_ms),
                }))
            else:
                console.print("[dim]ask something first[/dim]")
        else:
            try:
                with console.status("thinking..."):
                    last = runner.ask(question, history)
            except LLMUnavailableError as exc:
                console.print(f"[red]{exc}[/red]")
                continue
            _render(last)
            history.append((question, last.answer.answer or last.answer.clarifying_question or ""))


@app.command()
def intake(
    brief: Path = typer.Option(..., help="brief.txt"),
    docs: Path = typer.Option(..., help="Folder of documents"),
    out: Path = typer.Option(..., help="Run folder to write 00_intake/ and 01_guard/ into"),
    non_interactive: bool = typer.Option(False, "--non-interactive", help="Answer own questions, label ASSUMPTION"),
    config_path: Path = typer.Option(Path("forgeai.yaml"), "--config", help="forgeai.yaml"),
) -> None:
    """Stages 1-3: ingest, intake interview (<=5 questions) and groundability guard."""
    from forgeai.intake.orchestrate import run_intake_and_guard
    from forgeai.intake.schema import Question

    config = load_config(config_path)
    out.mkdir(parents=True, exist_ok=True)
    set_client(LLMClient(config, trace_path=out / "trace.jsonl"))
    preflight(config, out)

    def ask(question: Question) -> str:
        return typer.prompt(f"[{question.id}] {question.text}", default="", show_default=False)

    with run_lock(out, "intake"), stage_timer(out, "intake", config.min_free_ram_pct):
        req = run_intake_and_guard(
            brief.read_text(encoding="utf-8"), docs, out, config,
            SentenceTransformerEmbedder(config.embedding_model, config.embedding_revision),
            ask=None if non_interactive else ask,
        )
    console.print(f"[green]Requirements written to {out / '00_intake' / 'requirements.yaml'}[/green]")
    console.print(f"gaps: {[g.id for g in req.gaps] or 'none'}   assumptions: {len(req.assumptions)}")


@app.command()
def generate(
    run: Path = typer.Option(..., help="Run folder that already holds 00_intake/ and 01_guard/"),
    docs: Path = typer.Option(..., help="Folder of documents the agent will answer from"),
    config_path: Path = typer.Option(Path("forgeai.yaml"), "--config", help="forgeai.yaml"),
) -> None:
    """Stage 4: write 02_agent/agent.v0.yaml and tool_selection.json from the requirements."""
    from forgeai.generator.generate import generate_agent

    config = load_config(config_path)
    set_client(LLMClient(config, trace_path=run / "trace.jsonl"))
    preflight(config, run)
    with run_lock(run, "generate"), stage_timer(run, "generate", config.min_free_ram_pct):
        agent = generate_agent(run, docs, config)
    console.print(f"[green]Wrote {run / '02_agent' / 'agent.v0.yaml'}[/green]")
    console.print(f"tools: {agent.tools or 'none'}   min_score: {agent.retrieval.min_score} {agent.retrieval.min_score_bounds}")


@app.command()
def evals(
    run: Path = typer.Option(..., help="Run folder holding 00_intake/ (and 02_agent/ for the few-shot check)"),
    docs: Path = typer.Option(..., help="Folder of documents"),
    prefix: str = typer.Option("ev", help="Prefix for case ids"),
    config_path: Path = typer.Option(Path("forgeai.yaml"), "--config", help="forgeai.yaml"),
) -> None:
    """Stage 5: generate and validate the eval set into 03_evals/. Never reads the agent definition."""
    from forgeai.evals.build import build_eval_set, write_eval_set
    from forgeai.evals.corpus import EvalCorpus
    from forgeai.generator.fewshot import run_fewshot_check
    from forgeai.intake.schema import load_requirements

    config = load_config(config_path)
    set_client(LLMClient(config, trace_path=run / "trace.jsonl"))
    preflight(config, run)
    embedder = SentenceTransformerEmbedder(config.embedding_model, config.embedding_revision)  # loaded once per process
    req = load_requirements(run / "00_intake" / "requirements.yaml")
    brief_path = run / "00_intake" / "brief.txt"
    with run_lock(run, "evals"), stage_timer(run, "evals", config.min_free_ram_pct):
        result = build_eval_set(req, EvalCorpus.load(docs, config, embedder), config, embedder, prefix,
                                brief=brief_path.read_text(encoding="utf-8") if brief_path.is_file() else "")
        write_eval_set(result, run / "03_evals")
    stats = result.stats
    console.print(f"[green]{stats['valid_cases']} valid cases[/green] (floor {stats['floor']}, met: {stats['floor_met']}) "
                  f"selected {stats['selected']} split {stats['split']}")
    if (run / "02_agent" / "agent.v0.yaml").is_file():  # the build, not the eval generator, checks the few-shot examples
        check = run_fewshot_check(run, embedder, config.thresholds.fewshot_leakage_similarity,
                                  [c.question for c in result.cases])
        console.print(f"few-shot leakage check: {check['few_shot_examples_checked']} examples, {len(check['rejected'])} rejected")
    coverage = stats["coverage"]
    console.print(f"requirement coverage: {coverage['covered']}/{coverage['items']} items")
    if coverage["uncovered"]:
        console.print(f"[bold red]UNCOVERED requirement items (no eval case): {coverage['uncovered']}[/bold red] "
                      f"- see {run / '03_evals' / 'coverage_matrix.json'}")
        raise typer.Exit(3)


@app.command()
def audit(
    run: Path = typer.Option(..., help="Run folder holding 03_evals/eval_set.jsonl"),
    facts: Path = typer.Option(..., help="Hidden planted-facts file of a synthesized demo corpus"),
) -> None:
    """Audit the eval set against a hidden fact sheet (coverage and key-fact agreement)."""
    from forgeai.audit.audit import audit_eval_set
    from forgeai.evals.build import load_eval_set

    with run_lock(run, "audit"), stage_timer(run, "audit"):
        result = audit_eval_set(load_eval_set(run / "03_evals" / "eval_set.jsonl"), facts)
    (run / "03_evals" / "audit.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    console.print(f"coverage {result['coverage']['fully_covered']}/{result['coverage']['planted_facts']}  "
                  f"key-fact agreement {result['key_fact_agreement']['agree']}/{result['key_fact_agreement']['checked']}")


def _not_yet(phase: str) -> None:
    console.print(f"[yellow]Not implemented yet (arrives in phase {phase}).[/yellow]")
    raise typer.Exit(2)


@app.command()
def build() -> None:
    """Brief + documents -> tested agent (phases P3 to P9)."""
    _not_yet("P3-P9")


@app.command("eval")
def eval_() -> None:
    """Evaluate an agent (phase P5/P6)."""
    _not_yet("P5-P6")


@app.command()
def regress() -> None:
    """Regression mode (stretch, P10)."""
    _not_yet("P10")


@app.command()
def ui() -> None:
    """Streamlit UI (stretch, P10)."""
    _not_yet("P10")


if __name__ == "__main__":
    app()

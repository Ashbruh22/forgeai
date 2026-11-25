"""Command line interface: ``forgeai chat`` works now; other commands arrive with their phase."""

from __future__ import annotations

import json
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from forgeai.config import ConfigError, load_config
from forgeai.embeddings import SentenceTransformerEmbedder
from forgeai.llm import LLMClient, LLMUnavailableError, set_client
from forgeai.runner.agent_def import load_agent
from forgeai.runner.runner import AgentRunner, History, RunResult

app = typer.Typer(add_completion=False, help="ForgeAI: the agent that builds agents.")
console = Console()

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

    def ask(question: Question) -> str:
        return typer.prompt(f"[{question.id}] {question.text}", default="", show_default=False)

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
    agent = generate_agent(run, docs, config)
    console.print(f"[green]Wrote {run / '02_agent' / 'agent.v0.yaml'}[/green]")
    console.print(f"tools: {agent.tools or 'none'}   min_score: {agent.retrieval.min_score} {agent.retrieval.min_score_bounds}")


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

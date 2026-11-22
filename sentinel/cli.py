"""Sentinel CLI.

Implements the exact primary commands from the spec:
  sentinel -c | -t | -a <target> | -s | -f | -r | -resume <id>

Typer's `-x` short-flag style is emulated with explicit `--c/-c` style
boolean/callback options on a single-command app, since Typer/Click
model subcommands as words, not switches. We implement each as its own
Click-level option on the root callback to keep compatibility with the
literal requested syntax (`sentinel -c`, `sentinel -a target.htb`, ...).
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from sentinel.agent import Agent
from sentinel.config import get_settings
from sentinel.knowledge import KnowledgeBase
from sentinel.model_runtime import ModelRuntimeError, get_model_runtime
from sentinel.planner import build_context
from sentinel.policy import PolicyViolation
from sentinel.reports import generate_json_report, generate_markdown_report
from sentinel.schemas import AssessmentPhase, PlanStep
from sentinel.scope import ScopeError, ScopeManager
from sentinel.state_store import StateStore
from sentinel.tools.base import default_registry

app = typer.Typer(
    add_completion=False,
    help="Sentinel — local-first agentic penetration-testing assistant for authorized lab targets.",
)
console = Console()


knowledge_app = typer.Typer(help="Manage local security references used for retrieval augmented generation.")
app.add_typer(knowledge_app, name="knowledge")


@knowledge_app.command("ingest")
def knowledge_ingest(path: Optional[Path] = typer.Argument(None, help="A file or folder of PDF, Markdown, RST, or text references.")):
    settings = get_settings()
    path = path or settings.knowledge_dir
    if not path.exists():
        console.print(f"[red]Reference path does not exist: {path}[/red]")
        raise typer.Exit(code=1)
    db_path = settings.knowledge_db_path or settings.data_dir / "knowledge.db"
    result = KnowledgeBase(db_path).ingest(path)
    console.print(f"Indexed {result['chunks_added']} chunks from {result['files']} files; skipped {result['files_skipped']}.")
    for error in result["errors"]:
        console.print(f"[yellow]{error}[/yellow]")


@knowledge_app.command("search")
def knowledge_search(query: str = typer.Argument(...), limit: int = typer.Option(5, min=1, max=20)):
    settings = get_settings()
    db_path = settings.knowledge_db_path or settings.data_dir / "knowledge.db"
    matches = KnowledgeBase(db_path).search(query, limit=limit)
    for item in matches:
        console.print(f"[bold cyan][{item['source']}#{item['chunk']}][/bold cyan] {item['content']}")
    if not matches:
        console.print("No matching references found. Add material with 'sentinel knowledge ingest <path>'.")


def _context():
    settings = get_settings()
    store = StateStore(settings.db_path)
    registry = default_registry()
    scope = ScopeManager(store)
    return settings, store, registry, scope


def _resolve_assessment(store: StateStore, assessment_id: Optional[str]):
    if assessment_id:
        assessment = store.get_assessment(assessment_id)
        if assessment is None:
            console.print(f"[red]No such assessment: {assessment_id}[/red]")
            raise typer.Exit(code=1)
        return assessment
    assessment = store.latest_assessment()
    if assessment is None:
        console.print("[yellow]No assessments found. Start one with 'sentinel -a <target>'.[/yellow]")
        raise typer.Exit(code=1)
    return assessment


# --------------------------------------------------------------------------
# sentinel -c   Chat
# --------------------------------------------------------------------------
@app.command("chat", help="Ask the model about findings, lab concepts, SQL, Bash, or basic Kali tools.")
def chat_cmd(
    assessment_id: Optional[str] = typer.Option(None, "--assessment", "-id", help="Assessment to discuss."),
):
    settings, store, registry, scope = _context()
    try:
        model = get_model_runtime(settings)
    except Exception as exc:
        console.print(f"[red]Could not initialize model runtime:[/red] {exc}")
        raise typer.Exit(code=1)

    assessment = store.latest_assessment() if assessment_id is None else store.get_assessment(assessment_id)
    history: list[dict[str, str]] = []
    console.print("[bold]Sentinel chat.[/bold] Type 'exit' to quit. Ask about findings, lab concepts, "
                  "SQL, Bash, or basic Kali tools. Chat explains commands but does not execute them; "
                  "use 'sentinel -a <IP>' to start an assessment.")
    while True:
        try:
            user_input = typer.prompt(">")
        except (EOFError, KeyboardInterrupt):
            break
        if user_input.strip().lower() in {"exit", "quit"}:
            break
        context_note = ""
        if assessment is not None:
            ctx = build_context(assessment, store, registry)
            context_note = f"\n\n[Current assessment context: {ctx}]"
        history.append({"role": "user", "content": user_input + context_note})
        try:
            reply = model.chat(history)
        except ModelRuntimeError as exc:
            console.print(f"[red]Model error:[/red] {exc}")
            continue
        history.append({"role": "assistant", "content": reply})
        console.print(reply)


# --------------------------------------------------------------------------
# sentinel -t   Tools
# --------------------------------------------------------------------------
@app.command("tools", help="List registered security tools, install status, and operations.")
def tools_cmd():
    _, store, registry, _ = _context()
    table = Table(title="Registered Tools")
    table.add_column("Tool")
    table.add_column("Binary")
    table.add_column("Installed")
    table.add_column("Version")
    table.add_column("Tested OK")
    table.add_column("Operations")
    for tool in registry.all():
        status = tool.status()
        table.add_row(
            status.tool_name,
            status.binary,
            "[green]yes[/green]" if status.installed else "[red]no[/red]",
            status.version or "-",
            "[green]yes[/green]" if status.tested_ok else "[yellow]untested[/yellow]",
            ", ".join(status.operations),
        )
    console.print(table)
    console.print(
        "[dim]'Tested OK' reflects an actual prior successful invocation recorded against an "
        "assessment, not merely that the binary is present.[/dim]"
    )


# --------------------------------------------------------------------------
# sentinel -a <target>   Start assessment
# --------------------------------------------------------------------------
@app.command("assess", help="Start an assessment by IP; confirm authorization when prompted.")
def assess_cmd(
    target: str = typer.Argument(..., help="Lab machine IP or authorized hostname."),
    iterations: Optional[int] = typer.Option(None, "--max-iterations", help="Override the default iteration budget."),
    auto_approve: bool = typer.Option(
        False, "--auto-approve",
        help="Automatically approve intrusive actions (NOT recommended; use for scripted lab runs only).",
    ),
    rounds: int = typer.Option(1, "--rounds", help="How many plan/execute iterations to run now."),
):
    settings, store, registry, scope = _context()
    confirmed = typer.confirm(
        f"I confirm I am authorized to perform security testing against '{target}' "
        "(for example, my own machine or an HTB lab)."
    )
    if not confirmed:
        console.print("[yellow]Assessment cancelled.[/yellow]")
        raise typer.Exit(code=1)
    if not scope.check(target).allowed:
        try:
            scope.add(target, confirmed=True)
        except ScopeError as exc:
            console.print(f"[red]{exc}[/red]")
            raise typer.Exit(code=1)

    try:
        model = get_model_runtime(settings)
    except Exception as exc:
        console.print(f"[red]Could not initialize model runtime:[/red] {exc}")
        raise typer.Exit(code=1)

    agent = Agent(store, registry, scope, model, settings, console)
    try:
        assessment = agent.start_assessment(target, max_iterations=iterations)
    except ScopeError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Started assessment {assessment.assessment_id}[/bold green] for target {target}")

    def approval_cb(step: PlanStep) -> bool:
        action = step.proposed_action
        console.print(
            f"[yellow]Approval required:[/yellow] {action.tool_name}.{action.operation} — {action.rationale}"
        )
        return typer.confirm("Approve this intrusive action?", default=False)

    for _ in range(rounds):
        try:
            outcomes = agent.run_iteration(assessment, approval_cb=approval_cb, auto_approve=auto_approve)
        except PolicyViolation as exc:
            console.print(f"[red]{exc}[/red]")
            break
        for outcome in outcomes:
            action = outcome.step.proposed_action
            if outcome.executed:
                console.print(
                    f"  ran {action.tool_name}.{action.operation} "
                    f"(exit={outcome.exit_code}, timed_out={outcome.timed_out})"
                )
            else:
                console.print(f"  [dim]skipped {action.tool_name}.{action.operation}: {outcome.skipped_reason}[/dim]")

    console.print(f"Use 'sentinel -s' to check status or 'sentinel -r' to generate a report.")


# --------------------------------------------------------------------------
# sentinel -s   Status
# --------------------------------------------------------------------------
@app.command("status", help="Show the current or specified assessment's status.")
def status_cmd(assessment_id: Optional[str] = typer.Option(None, "--assessment", "-id")):
    _, store, registry, _ = _context()
    assessment = _resolve_assessment(store, assessment_id)

    services = store.list_services(assessment.assessment_id)
    findings = store.list_findings(assessment.assessment_id)
    invocations = store.list_invocations(assessment.assessment_id)
    pending = store.list_pending_approvals(assessment.assessment_id)

    console.print(f"[bold]Assessment {assessment.assessment_id}[/bold] — target {assessment.target}")
    console.print(f"  Phase: {assessment.phase.value}   Status: {assessment.status}")
    console.print(f"  Iterations: {assessment.iteration_count}/{assessment.max_iterations}")
    console.print(f"  Services discovered: {len(services)}")
    console.print(f"  Findings: {len(findings)}")
    console.print(f"  Tool invocations: {len(invocations)}")
    console.print(f"  Pending approvals: {len(pending)}")
    running_or_failed = [i for i in invocations if i.error or i.timed_out]
    if running_or_failed:
        console.print(f"  [yellow]Failed/timed-out invocations: {len(running_or_failed)}[/yellow]")
    if pending:
        for p in pending:
            console.print(f"    [yellow]Pending approval {p.approval_id}:[/yellow] {p.summary}")


# --------------------------------------------------------------------------
# sentinel -f   Findings
# --------------------------------------------------------------------------
@app.command("findings", help="Show findings for the selected assessment.")
def findings_cmd(assessment_id: Optional[str] = typer.Option(None, "--assessment", "-id")):
    _, store, _, _ = _context()
    assessment = _resolve_assessment(store, assessment_id)
    findings = store.list_findings(assessment.assessment_id)
    if not findings:
        console.print("[dim]No findings recorded yet.[/dim]")
        return
    table = Table(title=f"Findings — {assessment.target}")
    table.add_column("Severity")
    table.add_column("Title")
    table.add_column("Confidence")
    table.add_column("Status")
    table.add_column("Service")
    for f in findings:
        table.add_row(
            f.severity.value, f.title, f"{f.confidence:.2f}",
            f.verification_status.value, f.affected_service or "-",
        )
    console.print(table)


# --------------------------------------------------------------------------
# sentinel -r   Report
# --------------------------------------------------------------------------
@app.command("report", help="Generate a Markdown and/or JSON report for the selected assessment.")
def report_cmd(
    assessment_id: Optional[str] = typer.Option(None, "--assessment", "-id"),
    fmt: str = typer.Option("markdown", "--format", help="markdown | json | both"),
    out_dir: Optional[Path] = typer.Option(None, "--out", help="Output directory (defaults to configured reports dir)."),
):
    settings, store, _, _ = _context()
    assessment = _resolve_assessment(store, assessment_id)
    target_dir = out_dir or settings.reports_dir

    paths = []
    if fmt in ("markdown", "both"):
        paths.append(generate_markdown_report(assessment, store, target_dir))
    if fmt in ("json", "both"):
        paths.append(generate_json_report(assessment, store, target_dir))
    if not paths:
        console.print("[red]Unknown --format. Use markdown, json, or both.[/red]")
        raise typer.Exit(code=1)

    assessment.report_paths.extend(str(p) for p in paths)
    store.save_assessment(assessment)
    for p in paths:
        console.print(f"[green]Report written:[/green] {p}")


# --------------------------------------------------------------------------
# sentinel -resume <assessment_id>
# --------------------------------------------------------------------------
@app.command("resume", help="Resume a previously saved assessment.")
def resume_cmd(
    assessment_id: str = typer.Argument(...),
    rounds: int = typer.Option(1, "--rounds"),
    auto_approve: bool = typer.Option(False, "--auto-approve"),
):
    settings, store, registry, scope = _context()
    try:
        model = get_model_runtime(settings)
    except Exception as exc:
        console.print(f"[red]Could not initialize model runtime:[/red] {exc}")
        raise typer.Exit(code=1)

    agent = Agent(store, registry, scope, model, settings, console)
    try:
        assessment = agent.resume_assessment(assessment_id)
    except (ValueError, ScopeError) as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(code=1)

    console.print(f"[bold green]Resumed assessment {assessment.assessment_id}[/bold green] (target {assessment.target})")
    pending = store.list_pending_approvals(assessment.assessment_id)
    if pending:
        console.print(f"[yellow]{len(pending)} pending approval(s) from before the pause — re-review before continuing.[/yellow]")

    def approval_cb(step: PlanStep) -> bool:
        action = step.proposed_action
        console.print(f"[yellow]Approval required:[/yellow] {action.tool_name}.{action.operation} — {action.rationale}")
        return typer.confirm("Approve this intrusive action?", default=False)

    for _ in range(rounds):
        try:
            agent.run_iteration(assessment, approval_cb=approval_cb, auto_approve=auto_approve)
        except PolicyViolation as exc:
            console.print(f"[red]{exc}[/red]")
            break


# --------------------------------------------------------------------------
# Compatibility shim for the exact requested flag-style syntax:
#   sentinel -c | -t | -a <target> | -s | -f | -r | -resume <id>
# Typer/Click model subcommands as words; this rewrites argv before Click
# ever sees it so both styles work: `sentinel -c` and `sentinel chat`.
# --------------------------------------------------------------------------
_FLAG_TO_COMMAND = {
    "-c": "chat",
    "-t": "tools",
    "-a": "assess",
    "-s": "status",
    "-f": "findings",
    "-r": "report",
    "-resume": "resume",
}


def _rewrite_argv(argv: list[str]) -> list[str]:
    if not argv:
        return argv
    head, rest = argv[0], argv[1:]
    mapped = _FLAG_TO_COMMAND.get(head)
    if mapped is None:
        return argv
    return [mapped] + rest


def main() -> None:
    import sys

    sys.argv = [sys.argv[0]] + _rewrite_argv(sys.argv[1:])
    app()


if __name__ == "__main__":
    main()

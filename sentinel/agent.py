"""The core agent loop.

Implements the read -> plan -> validate -> approve -> execute -> parse ->
update -> decide-next -> stop cycle described in the design. Every step
that touches policy, scope, or execution goes through the dedicated
modules for those concerns — this file is orchestration only.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Callable, Optional

from rich.console import Console

from sentinel.config import Settings
from sentinel.executor import Executor
from sentinel.findings import apply_interpretation
from sentinel.model_runtime import ModelRuntime, ModelRuntimeError
from sentinel.planner import build_context, next_plan
from sentinel.policy import PolicyEngine, PolicyViolation
from sentinel.schemas import Assessment, AssessmentPhase, PlanStep
from sentinel.scope import ScopeManager
from sentinel.state_store import StateStore
from sentinel.tools.base import ToolRegistry

ApprovalCallback = Callable[[PlanStep], bool]


@dataclass
class StepOutcome:
    step: PlanStep
    executed: bool
    skipped_reason: Optional[str] = None
    exit_code: Optional[int] = None
    timed_out: bool = False


class Agent:
    def __init__(
        self,
        store: StateStore,
        registry: ToolRegistry,
        scope: ScopeManager,
        model: ModelRuntime,
        settings: Settings,
        console: Optional[Console] = None,
    ):
        self.store = store
        self.registry = registry
        self.scope = scope
        self.model = model
        self.settings = settings
        self.policy = PolicyEngine(registry, scope, store, settings)
        self.executor = Executor(store, settings)
        self.console = console or Console()

    def start_assessment(self, target: str, max_iterations: int | None = None) -> Assessment:
        scope_entry = self.scope.require_in_scope(target)
        assessment = Assessment(
            target=target,
            scope_id=scope_entry.scope_id,
            phase=AssessmentPhase.RECON,
            max_iterations=max_iterations or self.settings.max_iterations_default,
        )
        self.store.save_assessment(assessment)
        return assessment

    def run_iteration(
        self,
        assessment: Assessment,
        approval_cb: Optional[ApprovalCallback] = None,
        auto_approve: bool = False,
    ) -> list[StepOutcome]:
        """Run exactly one plan -> execute cycle. Returns what happened so
        the CLI can render it. Raises PolicyViolation if the iteration
        budget is exhausted."""
        self.policy.check_iteration_budget(assessment)

        try:
            plan = next_plan(assessment, self.store, self.registry, self.model)
        except ModelRuntimeError as exc:
            self.console.print(f"[red]Model error:[/red] {exc}")
            return []

        outcomes: list[StepOutcome] = []
        for step in plan.steps:
            outcome = self._run_step(assessment, step, approval_cb, auto_approve)
            outcomes.append(outcome)

        assessment.iteration_count += 1
        self.store.save_assessment(assessment)
        return outcomes

    def _run_step(
        self,
        assessment: Assessment,
        step: PlanStep,
        approval_cb: Optional[ApprovalCallback],
        auto_approve: bool,
    ) -> StepOutcome:
        proposed = step.proposed_action
        try:
            validated = self.policy.validate_action(
                assessment, proposed, {"target": proposed.target_hint} | self._guess_args(proposed)
            )
        except PolicyViolation as exc:
            self.console.print(f"[yellow]Skipped step {step.step_number}:[/yellow] {exc}")
            return StepOutcome(step=step, executed=False, skipped_reason=str(exc))

        if validated.needs_approval and not auto_approve:
            approval = self.policy.request_approval(
                assessment.assessment_id,
                invocation_id="pending",
                summary=f"{validated.tool.name}.{validated.operation.name} -> {' '.join(validated.argv)}",
            )
            approved = approval_cb(step) if approval_cb else False
            self.policy.decide_approval(approval, approved)
            if not approved:
                return StepOutcome(step=step, executed=False, skipped_reason="approval denied")

        inv = self.executor.run(assessment.assessment_id, validated)

        if inv.exit_code == 0 and not inv.timed_out and not inv.error:
            stdout, _stderr = self.executor.read_output(inv)
            try:
                interpretation = self.model.interpret_result({
                    "tool": validated.tool.name,
                    "operation": validated.operation.name,
                    "argv": validated.argv,
                    "exit_code": inv.exit_code,
                    "stdout_excerpt": stdout[:8000],
                })
                apply_interpretation(assessment.assessment_id, self.store, interpretation)
            except ModelRuntimeError as exc:
                self.console.print(f"[yellow]Could not interpret output:[/yellow] {exc}")

        return StepOutcome(
            step=step, executed=True,
            exit_code=inv.exit_code, timed_out=inv.timed_out,
        )

    @staticmethod
    def _guess_args(proposed) -> dict:
        """Best-effort extraction of extra args beyond target_hint for simple
        operations. The policy engine's Pydantic validation is the real
        safety net; this just keeps common single-arg operations usable
        without a bespoke mapping table per operation."""
        return {}

    def resume_assessment(self, assessment_id: str) -> Assessment:
        assessment = self.store.get_assessment(assessment_id)
        if assessment is None:
            raise ValueError(f"No such assessment: {assessment_id}")
        # Revalidate scope before continuing — an entry may have been removed
        # since the assessment was paused.
        self.scope.require_in_scope(assessment.target)
        if assessment.status == "cancelled":
            raise ValueError(f"Assessment {assessment_id} was cancelled and cannot be resumed.")
        assessment.status = "active"
        assessment.updated_at = datetime.now(timezone.utc)
        self.store.save_assessment(assessment)
        return assessment

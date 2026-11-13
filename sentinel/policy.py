"""Deterministic policy enforcement.

Everything in this module is plain Python logic — no LLM call ever
decides whether an action is permitted. The model can *propose* an
action; only this module (via the executor) decides whether it runs.
"""
from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, ValidationError

from sentinel.config import Settings
from sentinel.schemas import ActionRisk, Assessment, PendingApproval, ProposedAction
from sentinel.scope import ScopeError, ScopeManager
from sentinel.state_store import StateStore
from sentinel.tools.base import OperationSpec, ToolAdapter, ToolRegistry


class PolicyViolation(Exception):
    pass


@dataclass
class ValidatedAction:
    tool: ToolAdapter
    operation: OperationSpec
    args: BaseModel
    argv: list[str]
    risk: ActionRisk
    needs_approval: bool


class PolicyEngine:
    def __init__(
        self,
        registry: ToolRegistry,
        scope: ScopeManager,
        store: StateStore,
        settings: Settings,
    ):
        self.registry = registry
        self.scope = scope
        self.store = store
        self.settings = settings

    def validate_action(
        self,
        assessment: Assessment,
        proposed: ProposedAction,
        raw_args: dict,
    ) -> ValidatedAction:
        """Resolve a model-proposed action against the real tool registry,
        validate its arguments against the operation's Pydantic schema,
        and confirm the implied target is in authorized scope.

        Raises PolicyViolation on any failure. Never raises for things an
        operator should instead be asked to approve — that's a separate
        step (requires_approval / record_approval).
        """
        try:
            tool, op = self.registry.resolve(proposed.tool_name, proposed.operation)
        except KeyError as exc:
            raise PolicyViolation(str(exc)) from exc

        if not tool.is_installed():
            raise PolicyViolation(
                f"Tool '{tool.name}' is not installed or not on PATH. "
                f"Run 'sentinel -t' to check tool status."
            )

        try:
            args = op.args_model.model_validate(raw_args)
        except ValidationError as exc:
            raise PolicyViolation(f"Invalid arguments for {proposed.tool_name}.{proposed.operation}: {exc}") from exc

        target_field = self._extract_target(args)
        if target_field:
            try:
                self.scope.require_in_scope(target_field)
            except ScopeError as exc:
                raise PolicyViolation(str(exc)) from exc

        try:
            argv = op.build_argv(args)
        except Exception as exc:  # defensive: adapter bug should not crash the agent
            raise PolicyViolation(f"Failed to build command for {proposed.tool_name}.{proposed.operation}: {exc}") from exc

        if not argv or not isinstance(argv, list) or not all(isinstance(a, str) for a in argv):
            raise PolicyViolation("Adapter produced an invalid argv; refusing to execute.")

        effective_risk = op.risk
        if proposed.risk == ActionRisk.INTRUSIVE:
            effective_risk = ActionRisk.INTRUSIVE  # model can only escalate caution, never downgrade it

        needs_approval = (
            effective_risk == ActionRisk.INTRUSIVE and self.settings.require_approval_for_intrusive
        )

        return ValidatedAction(
            tool=tool, operation=op, args=args, argv=argv,
            risk=effective_risk, needs_approval=needs_approval,
        )

    @staticmethod
    def _extract_target(args: BaseModel) -> str | None:
        for field_name in ("target", "host", "url", "name"):
            if hasattr(args, field_name):
                value = getattr(args, field_name)
                if field_name == "url" and isinstance(value, str):
                    # strip scheme://host[:port] portion for scope checks
                    host = value.split("://", 1)[-1].split("/", 1)[0].split(":")[0]
                    return host
                if isinstance(value, str):
                    return value
        return None

    def request_approval(self, assessment_id: str, invocation_id: str, summary: str) -> PendingApproval:
        approval = PendingApproval(assessment_id=assessment_id, invocation_id=invocation_id, summary=summary)
        self.store.save_approval(approval)
        return approval

    def decide_approval(self, approval: PendingApproval, approved: bool) -> PendingApproval:
        from datetime import datetime, timezone

        approval.decided = True
        approval.approved = approved
        approval.decided_at = datetime.now(timezone.utc)
        self.store.save_approval(approval)
        return approval

    def check_iteration_budget(self, assessment: Assessment) -> None:
        if assessment.iteration_count >= assessment.max_iterations:
            raise PolicyViolation(
                f"Assessment {assessment.assessment_id} has reached its iteration limit "
                f"({assessment.max_iterations}). Resume will not proceed further without "
                f"an explicit budget increase."
            )

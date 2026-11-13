"""Typed, validated data structures shared across Sentinel.

All model-generated output and all tool output is parsed into these
Pydantic models before anything else in the system is allowed to act on it.
Nothing downstream trusts raw strings from the LLM or from subprocess output.
"""
from __future__ import annotations

import enum
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


def _now() -> datetime:
    return datetime.now(timezone.utc)


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


class AssessmentPhase(str, enum.Enum):
    CREATED = "created"
    RECON = "recon"
    ENUMERATION = "enumeration"
    INVESTIGATION = "investigation"
    VERIFICATION = "verification"
    REPORTING = "reporting"
    COMPLETE = "complete"
    CANCELLED = "cancelled"


class HypothesisStatus(str, enum.Enum):
    PROPOSED = "proposed"
    SUPPORTED = "supported"
    CONTRADICTED = "contradicted"
    VERIFIED = "verified"
    UNRESOLVED = "unresolved"


class FindingSeverity(str, enum.Enum):
    INFO = "info"
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class VerificationStatus(str, enum.Enum):
    UNVERIFIED_HYPOTHESIS = "unverified_hypothesis"
    SCANNER_ALERT = "scanner_alert"
    CONFIRMED = "confirmed"
    INFORMATIONAL = "informational"


class ActionRisk(str, enum.Enum):
    SAFE = "safe"          # passive / read-only, no approval needed
    INTRUSIVE = "intrusive"  # requires explicit human approval


class ScopeEntry(BaseModel):
    scope_id: str = Field(default_factory=lambda: new_id("scope"))
    target: str
    label: str = ""
    authorized_by_user: bool = False
    added_at: datetime = Field(default_factory=_now)
    notes: str = ""


class ToolInvocation(BaseModel):
    invocation_id: str = Field(default_factory=lambda: new_id("inv"))
    assessment_id: str
    tool_name: str
    operation: str
    args: list[str] = Field(default_factory=list)
    risk: ActionRisk = ActionRisk.SAFE
    approved: bool = False
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    exit_code: Optional[int] = None
    stdout_path: Optional[str] = None
    stderr_path: Optional[str] = None
    timed_out: bool = False
    error: Optional[str] = None


class Service(BaseModel):
    host: str
    port: int
    protocol: str = "tcp"
    service_name: Optional[str] = None
    product: Optional[str] = None
    version: Optional[str] = None
    extra_info: Optional[str] = None
    discovered_by: str = "nmap"
    discovered_at: datetime = Field(default_factory=_now)


class Hypothesis(BaseModel):
    hypothesis_id: str = Field(default_factory=lambda: new_id("hyp"))
    assessment_id: str
    description: str
    rationale: str = ""
    related_service: Optional[str] = None
    status: HypothesisStatus = HypothesisStatus.PROPOSED
    supporting_evidence: list[str] = Field(default_factory=list)
    contradicting_evidence: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)


class Finding(BaseModel):
    finding_id: str = Field(default_factory=lambda: new_id("find"))
    assessment_id: str
    title: str
    description: str
    severity: FindingSeverity = FindingSeverity.INFO
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    affected_service: Optional[str] = None
    verification_status: VerificationStatus = VerificationStatus.UNVERIFIED_HYPOTHESIS
    evidence_refs: list[str] = Field(default_factory=list)
    remediation: str = ""
    hypothesis_id: Optional[str] = None
    created_at: datetime = Field(default_factory=_now)

    @field_validator("confidence")
    @classmethod
    def _clamp(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class PendingApproval(BaseModel):
    approval_id: str = Field(default_factory=lambda: new_id("appr"))
    assessment_id: str
    invocation_id: str
    summary: str
    requested_at: datetime = Field(default_factory=_now)
    decided: bool = False
    approved: Optional[bool] = None
    decided_at: Optional[datetime] = None


class Assessment(BaseModel):
    assessment_id: str = Field(default_factory=lambda: new_id("assess"))
    target: str
    scope_id: str
    phase: AssessmentPhase = AssessmentPhase.CREATED
    status: str = "active"  # active | paused | cancelled | complete
    created_at: datetime = Field(default_factory=_now)
    updated_at: datetime = Field(default_factory=_now)
    completed_at: Optional[datetime] = None
    iteration_count: int = 0
    max_iterations: int = 40
    report_paths: list[str] = Field(default_factory=list)


# ---- Model-facing structured output -----------------------------------

class ProposedAction(BaseModel):
    """What the LLM proposes as the next step. Never executed directly;
    always re-validated by the policy engine against the tool registry
    and the argument schema for the named operation."""
    tool_name: str
    operation: str
    rationale: str
    target_hint: str = ""
    risk: ActionRisk = ActionRisk.SAFE


class PlanStep(BaseModel):
    step_number: int
    proposed_action: ProposedAction
    expected_evidence: str = ""


class Plan(BaseModel):
    assessment_id: str
    phase: AssessmentPhase
    steps: list[PlanStep] = Field(default_factory=list)
    summary: str = ""


class ModelInterpretation(BaseModel):
    """Structured result of asking the model to interpret raw tool output."""
    summary: str
    new_hypotheses: list[str] = Field(default_factory=list)
    supports_hypothesis_ids: list[str] = Field(default_factory=list)
    contradicts_hypothesis_ids: list[str] = Field(default_factory=list)
    suggested_follow_up: str = ""


# ---- Part 2: unified structured decision (fine-tuning target schema) ---

class StopCondition(str, enum.Enum):
    """Why the model is recommending the assessment stop, if it is."""
    NONE = "none"  # not recommending a stop; continue investigating
    OBJECTIVE_MET = "objective_met"
    SUFFICIENT_EVIDENCE = "sufficient_evidence"
    BUDGET_EXHAUSTED = "budget_exhausted"
    NO_VIABLE_NEXT_STEP = "no_viable_next_step"
    REQUIRES_USER_INPUT = "requires_user_input"
    SCOPE_VIOLATION_SUSPECTED = "scope_violation_suspected"


class DecisionProposedAction(BaseModel):
    """`proposed_action` in the Part 2 structured-decision contract. Kept as
    a distinct shape from `ProposedAction` because the fine-tuning target
    format (per the spec) nests raw argument key/values under `arguments`
    rather than a single `target_hint` string — the policy engine's
    Pydantic per-operation arg models are still what actually validates
    these values; this is only the model-facing envelope."""
    tool: str
    arguments: dict[str, Any] = Field(default_factory=dict)


class StructuredDecision(BaseModel):
    """The fine-tuning target / inference contract described in Master
    Prompt 2 §5. A single call that bundles observation, hypothesis
    tracking, the next proposed tool call, and a stop condition into one
    validated object. `StructuredDecision.to_plan_step()` and
    `.to_interpretation()` adapt it onto the schemas the existing
    agent loop (Part 1) already consumes, so the agent loop did not need
    to be rewritten to use this richer format.

    Nothing here authorizes execution: `proposed_action.tool` is only a
    string until `PolicyEngine.validate_action` resolves it against the
    real tool registry and re-validates `arguments` against that
    operation's own Pydantic schema, exactly as with `ProposedAction`.
    """
    assessment_id: str
    observations: list[str] = Field(default_factory=list)
    hypotheses: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    proposed_action: Optional[DecisionProposedAction] = None
    rationale: str = ""
    expected_evidence: list[str] = Field(default_factory=list)
    stop_condition: StopCondition = StopCondition.NONE

    def to_plan_step(self, step_number: int = 1) -> "PlanStep":
        """Adapt onto the Part 1 `PlanStep` shape so the existing agent
        loop can execute a decision without modification. Returns None
        (via caller check) when there's nothing to propose, e.g. the
        model recommended stopping."""
        action = self.proposed_action
        tool_name, operation = (action.tool.split(".", 1) + [""])[:2] if action else ("", "")
        proposed = ProposedAction(
            tool_name=tool_name,
            operation=operation,
            rationale=self.rationale,
            target_hint=str(action.arguments.get("target") or action.arguments.get("host") or action.arguments.get("url") or "") if action else "",
        )
        return PlanStep(
            step_number=step_number,
            proposed_action=proposed,
            expected_evidence="; ".join(self.expected_evidence),
        )

    def to_interpretation(self) -> "ModelInterpretation":
        """Adapt the observation/hypothesis side onto the Part 1
        `ModelInterpretation` shape consumed by `findings.apply_interpretation`."""
        return ModelInterpretation(
            summary="; ".join(self.observations) or self.rationale,
            new_hypotheses=list(self.hypotheses),
            suggested_follow_up="; ".join(self.missing_information),
        )

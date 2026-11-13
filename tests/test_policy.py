from __future__ import annotations

import pytest

from sentinel.config import Settings
from sentinel.policy import PolicyEngine, PolicyViolation
from sentinel.schemas import ActionRisk, Assessment, AssessmentPhase, ProposedAction
from sentinel.scope import ScopeManager
from sentinel.state_store import StateStore
from sentinel.tools.base import ToolRegistry


@pytest.fixture()
def policy(registry: ToolRegistry, scope: ScopeManager, store: StateStore, tmp_settings: Settings):
    return PolicyEngine(registry, scope, store, tmp_settings)


@pytest.fixture()
def in_scope_assessment(scope: ScopeManager, store: StateStore) -> Assessment:
    entry = scope.add("10.10.10.5", confirmed=True)
    assessment = Assessment(target="10.10.10.5", scope_id=entry.scope_id, phase=AssessmentPhase.RECON)
    store.save_assessment(assessment)
    return assessment


def test_rejects_unknown_tool(policy: PolicyEngine, in_scope_assessment: Assessment):
    proposed = ProposedAction(tool_name="nope", operation="scan", rationale="test")
    with pytest.raises(PolicyViolation):
        policy.validate_action(in_scope_assessment, proposed, {"target": "10.10.10.5"})


def test_rejects_unknown_operation(policy: PolicyEngine, in_scope_assessment: Assessment):
    proposed = ProposedAction(tool_name="nmap", operation="exploit", rationale="test")
    with pytest.raises(PolicyViolation):
        policy.validate_action(in_scope_assessment, proposed, {"target": "10.10.10.5"})


def test_rejects_out_of_scope_target(policy: PolicyEngine, in_scope_assessment: Assessment):
    proposed = ProposedAction(tool_name="nmap", operation="discover", rationale="test")
    with pytest.raises(PolicyViolation):
        policy.validate_action(in_scope_assessment, proposed, {"target": "8.8.8.8"})


def test_rejects_invalid_arguments(policy: PolicyEngine, in_scope_assessment: Assessment):
    proposed = ProposedAction(tool_name="nmap", operation="discover", rationale="test")
    with pytest.raises(PolicyViolation):
        policy.validate_action(in_scope_assessment, proposed, {"target": "10.10.10.5; whoami"})


def test_accepts_valid_safe_action(policy: PolicyEngine, in_scope_assessment: Assessment):
    proposed = ProposedAction(tool_name="nmap", operation="discover", rationale="test")
    validated = policy.validate_action(in_scope_assessment, proposed, {"target": "10.10.10.5"})
    assert validated.needs_approval is False
    assert validated.argv[0] == "nmap"


def test_intrusive_action_requires_approval(policy: PolicyEngine, in_scope_assessment: Assessment):
    proposed = ProposedAction(tool_name="http", operation="content_discovery", rationale="test")
    validated = policy.validate_action(
        in_scope_assessment, proposed, {"url": "http://10.10.10.5/"}
    )
    assert validated.needs_approval is True


def test_model_cannot_downgrade_risk(policy: PolicyEngine, in_scope_assessment: Assessment):
    # content_discovery is INTRUSIVE at the operation level; even if the model
    # labels its proposal "safe", the operation's own risk still governs.
    proposed = ProposedAction(
        tool_name="http", operation="content_discovery", rationale="test", risk=ActionRisk.SAFE
    )
    validated = policy.validate_action(in_scope_assessment, proposed, {"url": "http://10.10.10.5/"})
    assert validated.risk == ActionRisk.INTRUSIVE


def test_iteration_budget_enforced(policy: PolicyEngine, in_scope_assessment: Assessment):
    in_scope_assessment.max_iterations = 1
    in_scope_assessment.iteration_count = 1
    with pytest.raises(PolicyViolation):
        policy.check_iteration_budget(in_scope_assessment)


def test_nuclei_rejects_unapproved_tags(policy: PolicyEngine, in_scope_assessment: Assessment):
    proposed = ProposedAction(tool_name="nuclei", operation="scan_approved_templates", rationale="test")
    with pytest.raises(PolicyViolation):
        policy.validate_action(
            in_scope_assessment, proposed,
            {"url": "http://10.10.10.5/", "tags": "exploits,rce"},
        )

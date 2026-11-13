from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "training"))

import evaluate_model  # noqa: E402

from sentinel.config import Settings
from sentinel.policy import PolicyEngine
from sentinel.schemas import DecisionProposedAction, StopCondition, StructuredDecision
from sentinel.scope import ScopeManager
from sentinel.state_store import StateStore
from sentinel.tools.base import default_registry


def _case(scenario_id, context, label_dict):
    return {
        "context": context,
        "label": StructuredDecision.model_validate(label_dict),
        "meta": {"scenario_id": scenario_id, "category": "test"},
    }


def _setup(tmp_settings: Settings):
    store = StateStore(tmp_settings.db_path)
    scope = ScopeManager(store)
    registry = default_registry()
    policy = PolicyEngine(registry, scope, store, tmp_settings)
    return store, scope, registry, policy


def test_perfect_decider_scores_well_on_non_tool_metrics(tmp_settings: Settings):
    store, scope, registry, policy = _setup(tmp_settings)
    cases = [
        _case("s1", {"assessment_id": "a1"}, {
            "assessment_id": "a1", "observations": [], "stop_condition": "objective_met",
        }),
        _case("s2", {"assessment_id": "a2"}, {
            "assessment_id": "a2", "observations": [], "stop_condition": "sufficient_evidence",
        }),
    ]
    assessment = evaluate_model._build_eval_assessment(store, scope, cases)
    decider = evaluate_model._perfect_decider_factory(cases)
    report = evaluate_model.evaluate(decider, cases, "perfect", policy, assessment)

    assert report.structured_output_validity == 1.0
    assert report.stop_condition_accuracy == 1.0
    assert report.tool_selection_accuracy == 1.0  # both None == None


def test_tool_match_requires_identical_tool_string(tmp_settings: Settings):
    store, scope, registry, policy = _setup(tmp_settings)
    label = {
        "assessment_id": "a1",
        "proposed_action": {"tool": "nmap.discover", "arguments": {"target": "10.10.10.5"}},
        "stop_condition": "none",
    }
    cases = [_case("s1", {"assessment_id": "a1"}, label)]
    assessment = evaluate_model._build_eval_assessment(store, scope, cases)

    def wrong_tool_decider(context):
        return StructuredDecision(
            assessment_id="a1",
            proposed_action=DecisionProposedAction(tool="nmap.service_scan", arguments={"target": "10.10.10.5"}),
            stop_condition=StopCondition.NONE,
        )

    report = evaluate_model.evaluate(wrong_tool_decider, cases, "wrong", policy, assessment)
    assert report.tool_selection_accuracy == 0.0


def test_duplicate_action_detected_against_completed_operations(tmp_settings: Settings):
    store, scope, registry, policy = _setup(tmp_settings)
    scope.add("10.10.10.5", confirmed=True)
    context = {
        "assessment_id": "a1",
        "completed_operations": [{"tool": "nmap", "operation": "discover", "exit_code": 0}],
    }
    cases = [_case("s1", context, {"assessment_id": "a1", "stop_condition": "none"})]
    assessment = evaluate_model._build_eval_assessment(store, scope, cases)

    def repeat_decider(ctx):
        return StructuredDecision(
            assessment_id="a1",
            proposed_action=DecisionProposedAction(tool="nmap.discover", arguments={"target": "10.10.10.5"}),
            stop_condition=StopCondition.NONE,
        )

    report = evaluate_model.evaluate(repeat_decider, cases, "repeat", policy, assessment)
    assert report.duplicate_action_rate == 1.0


def test_decider_that_raises_counts_as_invalid_structured_output(tmp_settings: Settings):
    store, scope, registry, policy = _setup(tmp_settings)
    cases = [_case("s1", {"assessment_id": "a1"}, {"assessment_id": "a1", "stop_condition": "none"})]
    assessment = evaluate_model._build_eval_assessment(store, scope, cases)

    def broken(ctx):
        raise ValueError("model returned unparseable text")

    report = evaluate_model.evaluate(broken, cases, "broken", policy, assessment)
    assert report.structured_output_validity == 0.0
    assert len(report.failed_cases) == 1
    assert report.failed_cases[0]["error"] == "model returned unparseable text"


def test_unsupported_claim_rate_flags_fabricated_observation():
    context = {"stdout_excerpt": "PORT 22/tcp open ssh"}
    rate = evaluate_model._unsupported_claim_rate(
        ["a completely unrelated claim about zebras migrating"], context
    )
    assert rate == 1.0


def test_unsupported_claim_rate_accepts_grounded_observation():
    context = {"stdout_excerpt": "PORT 22/tcp open ssh OpenSSH 8.2p1"}
    rate = evaluate_model._unsupported_claim_rate(
        ["OpenSSH 8.2p1 is running on the open ssh port"], context
    )
    assert rate == 0.0


def test_collect_referenced_targets_pulls_from_context_and_label():
    cases = [
        _case("s1", {"assessment_id": "a1", "target": "10.10.10.9", "services": [{"host": "10.10.10.9"}]}, {
            "assessment_id": "a1",
            "proposed_action": {"tool": "http.fetch_headers", "arguments": {"url": "http://10.10.10.10/"}},
            "stop_condition": "none",
        }),
    ]
    targets = evaluate_model._collect_referenced_targets(cases)
    assert "10.10.10.9" in targets
    assert "10.10.10.10" in targets

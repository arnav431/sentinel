from __future__ import annotations

import pytest
from pydantic import ValidationError

from sentinel.schemas import (
    DecisionProposedAction,
    StopCondition,
    StructuredDecision,
)


def test_structured_decision_minimal_valid():
    decision = StructuredDecision(assessment_id="assess_1")
    assert decision.stop_condition == StopCondition.NONE
    assert decision.proposed_action is None


def test_structured_decision_rejects_bad_stop_condition():
    with pytest.raises(ValidationError):
        StructuredDecision(assessment_id="assess_1", stop_condition="give_up")


def test_structured_decision_round_trips_through_json():
    decision = StructuredDecision(
        assessment_id="assess_1",
        observations=["port 80 is open"],
        hypotheses=["may be running an old CMS"],
        proposed_action=DecisionProposedAction(tool="nmap.discover", arguments={"target": "10.10.10.5"}),
        rationale="cheapest next step",
        expected_evidence=["open port list"],
        stop_condition=StopCondition.NONE,
    )
    raw = decision.model_dump_json()
    reloaded = StructuredDecision.model_validate_json(raw)
    assert reloaded == decision


def test_to_plan_step_splits_tool_and_operation():
    decision = StructuredDecision(
        assessment_id="assess_1",
        proposed_action=DecisionProposedAction(tool="nmap.discover", arguments={"target": "10.10.10.5"}),
        rationale="r",
        expected_evidence=["x"],
    )
    step = decision.to_plan_step(step_number=3)
    assert step.step_number == 3
    assert step.proposed_action.tool_name == "nmap"
    assert step.proposed_action.operation == "discover"
    assert step.proposed_action.target_hint == "10.10.10.5"
    assert step.expected_evidence == "x"


def test_to_plan_step_handles_no_action():
    decision = StructuredDecision(assessment_id="assess_1", stop_condition=StopCondition.OBJECTIVE_MET)
    step = decision.to_plan_step()
    assert step.proposed_action.tool_name == ""
    assert step.proposed_action.operation == ""


def test_to_interpretation_carries_observations_and_hypotheses():
    decision = StructuredDecision(
        assessment_id="assess_1",
        observations=["banner shows OpenSSH 8.2p1"],
        hypotheses=["may support legacy key exchange algorithms"],
        missing_information=["confirm via ssh2-enum-algos"],
    )
    interpretation = decision.to_interpretation()
    assert "banner shows OpenSSH 8.2p1" in interpretation.summary
    assert interpretation.new_hypotheses == ["may support legacy key exchange algorithms"]
    assert "confirm via ssh2-enum-algos" in interpretation.suggested_follow_up


def test_decision_proposed_action_defaults_empty_arguments():
    action = DecisionProposedAction(tool="dns.lookup")
    assert action.arguments == {}

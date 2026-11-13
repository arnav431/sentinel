"""Planning layer.

Builds the context handed to the model and turns the model's structured
`Plan` into a queue of `ProposedAction`s for the agent loop. Deciding
which operations even *exist* to propose comes from the tool registry,
not from the model's imagination.
"""
from __future__ import annotations

from typing import Any

from sentinel.model_runtime import ModelRuntime
from sentinel.schemas import Assessment, Plan
from sentinel.state_store import StateStore
from sentinel.tools.base import ToolRegistry


def build_context(assessment: Assessment, store: StateStore, registry: ToolRegistry) -> dict[str, Any]:
    services = store.list_services(assessment.assessment_id)
    hypotheses = store.list_hypotheses(assessment.assessment_id)
    findings = store.list_findings(assessment.assessment_id)
    invocations = store.list_invocations(assessment.assessment_id)

    available_operations = []
    for tool in registry.all():
        if not tool.is_installed():
            continue
        for op in tool.operations().values():
            available_operations.append({
                "tool_name": tool.name,
                "operation": op.name,
                "description": op.description,
                "risk": op.risk.value,
            })

    return {
        "assessment_id": assessment.assessment_id,
        "target": assessment.target,
        "phase": assessment.phase.value,
        "iteration_count": assessment.iteration_count,
        "max_iterations": assessment.max_iterations,
        "services": [s.model_dump(mode="json") for s in services],
        "hypotheses": [h.model_dump(mode="json") for h in hypotheses],
        "findings": [f.model_dump(mode="json") for f in findings],
        "completed_operations": [
            {"tool": i.tool_name, "operation": i.operation, "exit_code": i.exit_code}
            for i in invocations
        ],
        "available_operations": available_operations,
    }


def next_plan(
    assessment: Assessment,
    store: StateStore,
    registry: ToolRegistry,
    model: ModelRuntime,
) -> Plan:
    context = build_context(assessment, store, registry)
    return model.create_plan(context)

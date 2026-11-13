"""Finding construction and hypothesis bookkeeping.

Encodes the rule that a finding's verification status can only move
toward "confirmed" through an explicit, recorded verification step —
never purely because the model said so.
"""
from __future__ import annotations

from sentinel.schemas import (
    Finding,
    FindingSeverity,
    Hypothesis,
    HypothesisStatus,
    ModelInterpretation,
    VerificationStatus,
)
from sentinel.state_store import StateStore


def apply_interpretation(
    assessment_id: str,
    store: StateStore,
    interpretation: ModelInterpretation,
    related_service: str | None = None,
) -> list[Hypothesis]:
    """Persist new hypotheses and update status of existing ones based on a
    model interpretation of tool output. Never creates a Finding directly —
    that requires a separate, explicit verification step."""
    created: list[Hypothesis] = []

    for desc in interpretation.new_hypotheses:
        hyp = Hypothesis(
            assessment_id=assessment_id,
            description=desc,
            rationale=interpretation.summary,
            related_service=related_service,
        )
        store.save_hypothesis(hyp)
        created.append(hyp)

    existing = {h.hypothesis_id: h for h in store.list_hypotheses(assessment_id)}
    for hid in interpretation.supports_hypothesis_ids:
        hyp = existing.get(hid)
        if hyp:
            hyp.status = HypothesisStatus.SUPPORTED
            hyp.supporting_evidence.append(interpretation.summary)
            store.save_hypothesis(hyp)
    for hid in interpretation.contradicts_hypothesis_ids:
        hyp = existing.get(hid)
        if hyp:
            hyp.status = HypothesisStatus.CONTRADICTED
            hyp.contradicting_evidence.append(interpretation.summary)
            store.save_hypothesis(hyp)

    return created


def record_finding(
    store: StateStore,
    assessment_id: str,
    title: str,
    description: str,
    severity: FindingSeverity,
    confidence: float,
    verification_status: VerificationStatus,
    affected_service: str | None = None,
    evidence_refs: list[str] | None = None,
    remediation: str = "",
    hypothesis_id: str | None = None,
) -> Finding:
    finding = Finding(
        assessment_id=assessment_id,
        title=title,
        description=description,
        severity=severity,
        confidence=confidence,
        affected_service=affected_service,
        verification_status=verification_status,
        evidence_refs=evidence_refs or [],
        remediation=remediation,
        hypothesis_id=hypothesis_id,
    )
    store.save_finding(finding)

    if hypothesis_id and verification_status == VerificationStatus.CONFIRMED:
        hyps = {h.hypothesis_id: h for h in store.list_hypotheses(assessment_id)}
        hyp = hyps.get(hypothesis_id)
        if hyp:
            hyp.status = HypothesisStatus.VERIFIED
            store.save_hypothesis(hyp)

    return finding

from __future__ import annotations

import json

from sentinel.reports import generate_json_report, generate_markdown_report
from sentinel.schemas import Assessment, AssessmentPhase, Finding, FindingSeverity, VerificationStatus
from sentinel.state_store import StateStore


def _assessment_with_finding(store: StateStore) -> Assessment:
    a = Assessment(target="10.10.10.5", scope_id="scope_x", phase=AssessmentPhase.REPORTING)
    store.save_assessment(a)
    f = Finding(
        assessment_id=a.assessment_id,
        title="Anonymous FTP login",
        description="FTP allows anonymous access.",
        severity=FindingSeverity.MEDIUM,
        confidence=0.9,
        verification_status=VerificationStatus.CONFIRMED,
        remediation="Disable anonymous FTP access.",
    )
    store.save_finding(f)
    return a


def test_markdown_report_contains_finding(store: StateStore, tmp_settings):
    a = _assessment_with_finding(store)
    path = generate_markdown_report(a, store, tmp_settings.reports_dir)
    text = path.read_text()
    assert "Anonymous FTP login" in text
    assert "Disable anonymous FTP access" in text
    assert a.assessment_id in text


def test_json_report_round_trips(store: StateStore, tmp_settings):
    a = _assessment_with_finding(store)
    path = generate_json_report(a, store, tmp_settings.reports_dir)
    data = json.loads(path.read_text())
    assert data["assessment"]["assessment_id"] == a.assessment_id
    assert len(data["findings"]) == 1
    assert data["findings"][0]["title"] == "Anonymous FTP login"

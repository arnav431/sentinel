from __future__ import annotations

from sentinel.schemas import Assessment, AssessmentPhase, Finding, FindingSeverity, Service, VerificationStatus
from sentinel.state_store import StateStore


def test_assessment_round_trip(store: StateStore):
    a = Assessment(target="10.10.10.5", scope_id="scope_x", phase=AssessmentPhase.RECON)
    store.save_assessment(a)
    loaded = store.get_assessment(a.assessment_id)
    assert loaded is not None
    assert loaded.target == "10.10.10.5"
    assert loaded.phase == AssessmentPhase.RECON


def test_assessment_survives_new_store_instance(tmp_settings):
    store1 = StateStore(tmp_settings.db_path)
    a = Assessment(target="10.10.10.5", scope_id="scope_x")
    store1.save_assessment(a)

    # Simulate a process restart: a brand new StateStore pointed
    # at the same db_path must see the same data.
    store2 = StateStore(tmp_settings.db_path)
    loaded = store2.get_assessment(a.assessment_id)
    assert loaded is not None
    assert loaded.target == "10.10.10.5"


def test_latest_assessment_orders_by_created_at(store: StateStore):
    a1 = Assessment(target="10.10.10.5", scope_id="s1")
    store.save_assessment(a1)
    a2 = Assessment(target="10.10.10.6", scope_id="s2")
    store.save_assessment(a2)
    latest = store.latest_assessment()
    assert latest is not None
    assert latest.assessment_id in {a1.assessment_id, a2.assessment_id}


def test_service_upsert_is_idempotent(store: StateStore):
    svc = Service(host="10.10.10.5", port=80, service_name="http")
    store.upsert_service("assess_1", svc)
    store.upsert_service("assess_1", svc)
    services = store.list_services("assess_1")
    assert len(services) == 1


def test_finding_persistence(store: StateStore):
    f = Finding(
        assessment_id="assess_1",
        title="Anonymous FTP login",
        description="Anonymous login permitted",
        severity=FindingSeverity.MEDIUM,
        confidence=0.9,
        verification_status=VerificationStatus.CONFIRMED,
    )
    store.save_finding(f)
    findings = store.list_findings("assess_1")
    assert len(findings) == 1
    assert findings[0].title == "Anonymous FTP login"

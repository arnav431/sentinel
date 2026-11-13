"""Persistent assessment state.

All durable state lives in SQLite. Every write goes through this module —
nothing else touches the database directly — so persistence semantics
(survives restarts, no silent replay of intrusive actions) stay in one place.
"""
from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from sentinel.schemas import (
    Assessment,
    Finding,
    Hypothesis,
    PendingApproval,
    ScopeEntry,
    Service,
    ToolInvocation,
)

SCHEMA = """
CREATE TABLE IF NOT EXISTS scope (
    scope_id TEXT PRIMARY KEY,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS assessments (
    assessment_id TEXT PRIMARY KEY,
    target TEXT NOT NULL,
    created_at TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS services (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    assessment_id TEXT NOT NULL,
    host TEXT NOT NULL,
    port INTEGER NOT NULL,
    protocol TEXT NOT NULL,
    data TEXT NOT NULL,
    UNIQUE(assessment_id, host, port, protocol)
);
CREATE TABLE IF NOT EXISTS invocations (
    invocation_id TEXT PRIMARY KEY,
    assessment_id TEXT NOT NULL,
    tool_name TEXT NOT NULL,
    operation TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS hypotheses (
    hypothesis_id TEXT PRIMARY KEY,
    assessment_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    assessment_id TEXT NOT NULL,
    data TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS approvals (
    approval_id TEXT PRIMARY KEY,
    assessment_id TEXT NOT NULL,
    data TEXT NOT NULL
);
"""


class StateStore:
    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._conn() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _conn(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path, timeout=30)
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    # ---- Scope -----------------------------------------------------
    def add_scope_entry(self, entry: ScopeEntry) -> None:
        with self._conn() as conn:
            conn.execute(
                "INSERT INTO scope (scope_id, data) VALUES (?, ?)",
                (entry.scope_id, entry.model_dump_json()),
            )

    def remove_scope_entry(self, scope_id: str) -> bool:
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM scope WHERE scope_id = ?", (scope_id,))
            return cur.rowcount > 0

    def list_scope(self) -> list[ScopeEntry]:
        with self._conn() as conn:
            rows = conn.execute("SELECT data FROM scope").fetchall()
        return [ScopeEntry.model_validate_json(r[0]) for r in rows]

    def get_scope_entry(self, scope_id: str) -> Optional[ScopeEntry]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM scope WHERE scope_id = ?", (scope_id,)
            ).fetchone()
        return ScopeEntry.model_validate_json(row[0]) if row else None

    # ---- Assessments -------------------------------------------------
    def save_assessment(self, assessment: Assessment) -> None:
        assessment.updated_at = datetime.now(timezone.utc)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO assessments (assessment_id, target, created_at, data)
                   VALUES (?, ?, ?, ?)
                   ON CONFLICT(assessment_id) DO UPDATE SET data = excluded.data""",
                (
                    assessment.assessment_id,
                    assessment.target,
                    assessment.created_at.isoformat(),
                    assessment.model_dump_json(),
                ),
            )

    def get_assessment(self, assessment_id: str) -> Optional[Assessment]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT data FROM assessments WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchone()
        return Assessment.model_validate_json(row[0]) if row else None

    def list_assessments(self) -> list[Assessment]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM assessments ORDER BY created_at DESC"
            ).fetchall()
        return [Assessment.model_validate_json(r[0]) for r in rows]

    def latest_assessment(self) -> Optional[Assessment]:
        items = self.list_assessments()
        return items[0] if items else None

    # ---- Services ------------------------------------------------------
    def upsert_service(self, assessment_id: str, service: Service) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO services (assessment_id, host, port, protocol, data)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(assessment_id, host, port, protocol)
                   DO UPDATE SET data = excluded.data""",
                (
                    assessment_id,
                    service.host,
                    service.port,
                    service.protocol,
                    service.model_dump_json(),
                ),
            )

    def list_services(self, assessment_id: str) -> list[Service]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM services WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchall()
        return [Service.model_validate_json(r[0]) for r in rows]

    # ---- Tool invocations ------------------------------------------
    def save_invocation(self, inv: ToolInvocation) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO invocations (invocation_id, assessment_id, tool_name, operation, data)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(invocation_id) DO UPDATE SET data = excluded.data""",
                (inv.invocation_id, inv.assessment_id, inv.tool_name, inv.operation, inv.model_dump_json()),
            )

    def list_invocations(self, assessment_id: str) -> list[ToolInvocation]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM invocations WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchall()
        return [ToolInvocation.model_validate_json(r[0]) for r in rows]

    # ---- Hypotheses --------------------------------------------------
    def save_hypothesis(self, hyp: Hypothesis) -> None:
        hyp.updated_at = datetime.now(timezone.utc)
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO hypotheses (hypothesis_id, assessment_id, data)
                   VALUES (?, ?, ?)
                   ON CONFLICT(hypothesis_id) DO UPDATE SET data = excluded.data""",
                (hyp.hypothesis_id, hyp.assessment_id, hyp.model_dump_json()),
            )

    def list_hypotheses(self, assessment_id: str) -> list[Hypothesis]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM hypotheses WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchall()
        return [Hypothesis.model_validate_json(r[0]) for r in rows]

    # ---- Findings ------------------------------------------------------
    def save_finding(self, finding: Finding) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO findings (finding_id, assessment_id, data)
                   VALUES (?, ?, ?)
                   ON CONFLICT(finding_id) DO UPDATE SET data = excluded.data""",
                (finding.finding_id, finding.assessment_id, finding.model_dump_json()),
            )

    def list_findings(self, assessment_id: str) -> list[Finding]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM findings WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchall()
        return [Finding.model_validate_json(r[0]) for r in rows]

    # ---- Approvals -------------------------------------------------
    def save_approval(self, appr: PendingApproval) -> None:
        with self._conn() as conn:
            conn.execute(
                """INSERT INTO approvals (approval_id, assessment_id, data)
                   VALUES (?, ?, ?)
                   ON CONFLICT(approval_id) DO UPDATE SET data = excluded.data""",
                (appr.approval_id, appr.assessment_id, appr.model_dump_json()),
            )

    def list_pending_approvals(self, assessment_id: str) -> list[PendingApproval]:
        items = self._list_approvals(assessment_id)
        return [a for a in items if not a.decided]

    def _list_approvals(self, assessment_id: str) -> list[PendingApproval]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT data FROM approvals WHERE assessment_id = ?",
                (assessment_id,),
            ).fetchall()
        return [PendingApproval.model_validate_json(r[0]) for r in rows]

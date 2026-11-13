"""Execution layer.

Runs only already-validated argv lists produced by the policy engine.
Never constructs a shell string, never passes shell=True, never takes a
command from the model directly. Captures stdout/stderr to evidence
files, enforces a timeout and an output-size cap, and persists the
invocation record regardless of outcome.
"""
from __future__ import annotations

import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

from sentinel.config import Settings
from sentinel.policy import ValidatedAction
from sentinel.schemas import ToolInvocation
from sentinel.state_store import StateStore


class Executor:
    def __init__(self, store: StateStore, settings: Settings):
        self.store = store
        self.settings = settings

    def run(self, assessment_id: str, validated: ValidatedAction) -> ToolInvocation:
        inv = ToolInvocation(
            assessment_id=assessment_id,
            tool_name=validated.tool.name,
            operation=validated.operation.name,
            args=validated.argv,
            risk=validated.risk,
            approved=not validated.needs_approval,
        )
        inv.started_at = datetime.now(timezone.utc)
        self.store.save_invocation(inv)

        evidence_dir = self.settings.evidence_dir / assessment_id
        evidence_dir.mkdir(parents=True, exist_ok=True)
        stdout_path = evidence_dir / f"{inv.invocation_id}.stdout.txt"
        stderr_path = evidence_dir / f"{inv.invocation_id}.stderr.txt"

        try:
            proc = subprocess.run(
                validated.argv,
                capture_output=True,
                text=True,
                timeout=self.settings.tool_timeout_s,
                shell=False,
            )
            stdout = proc.stdout[: self.settings.max_output_bytes]
            stderr = proc.stderr[: self.settings.max_output_bytes]
            stdout_path.write_text(stdout, encoding="utf-8", errors="replace")
            stderr_path.write_text(stderr, encoding="utf-8", errors="replace")
            inv.exit_code = proc.returncode
            inv.stdout_path = str(stdout_path)
            inv.stderr_path = str(stderr_path)
        except subprocess.TimeoutExpired as exc:
            inv.timed_out = True
            inv.error = f"timed out after {self.settings.tool_timeout_s}s"
            partial_out = (exc.stdout or "")[: self.settings.max_output_bytes] if isinstance(exc.stdout, str) else ""
            partial_err = (exc.stderr or "")[: self.settings.max_output_bytes] if isinstance(exc.stderr, str) else ""
            stdout_path.write_text(partial_out, encoding="utf-8", errors="replace")
            stderr_path.write_text(partial_err, encoding="utf-8", errors="replace")
            inv.stdout_path = str(stdout_path)
            inv.stderr_path = str(stderr_path)
        except FileNotFoundError as exc:
            inv.error = f"executable not found: {exc}"
        except Exception as exc:  # last-resort: never let a tool crash the agent loop
            inv.error = f"execution failed: {exc}"
        finally:
            inv.completed_at = datetime.now(timezone.utc)
            self.store.save_invocation(inv)

        return inv

    def read_output(self, inv: ToolInvocation) -> tuple[str, str]:
        stdout = Path(inv.stdout_path).read_text(encoding="utf-8", errors="replace") if inv.stdout_path else ""
        stderr = Path(inv.stderr_path).read_text(encoding="utf-8", errors="replace") if inv.stderr_path else ""
        return stdout, stderr

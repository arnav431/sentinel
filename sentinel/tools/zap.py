from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

from sentinel.schemas import ActionRisk
from sentinel.tools.base import OperationSpec, ToolAdapter

_URL_RE = re.compile(r"^https?://[A-Za-z0-9.\-]+(:\d+)?(/.*)?$")


class PassiveScanArgs(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def _vu(cls, v: str) -> str:
        v = v.strip()
        if not _URL_RE.match(v):
            raise ValueError(f"invalid URL: {v!r}")
        return v


class ZapTool(ToolAdapter):
    """OWASP ZAP passive assessment via the zap-baseline.py wrapper script,
    which crawls and passively analyzes traffic without active attack
    payloads."""

    name = "zap"
    binary = "zap-baseline.py"

    def register_operations(self) -> None:
        self._op(OperationSpec(
            name="passive_baseline",
            description="Run the ZAP baseline passive scan (spider + passive rules only, "
                        "no active attack payloads) and emit a JSON report.",
            args_model=PassiveScanArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: [
                "zap-baseline.py", "-t", a.url, "-J", "/tmp/zap-report.json", "-I",
            ],
        ))

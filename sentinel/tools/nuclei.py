from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

from sentinel.schemas import ActionRisk
from sentinel.tools.base import OperationSpec, ToolAdapter

_URL_RE = re.compile(r"^https?://[A-Za-z0-9.\-]+(:\d+)?(/.*)?$")

# Only these template tags may ever be selected. This list is deliberately
# conservative (informational / low-risk checks) and is enforced in code,
# not left to the model to choose freely.
APPROVED_TAGS = {"tech", "exposures", "misconfiguration", "default-login", "cve-info"}


class NucleiScanArgs(BaseModel):
    url: str
    tags: str = "tech,exposures"

    @field_validator("url")
    @classmethod
    def _vu(cls, v: str) -> str:
        v = v.strip()
        if not _URL_RE.match(v):
            raise ValueError(f"invalid URL: {v!r}")
        return v

    @field_validator("tags")
    @classmethod
    def _vt(cls, v: str) -> str:
        requested = {t.strip() for t in v.split(",") if t.strip()}
        if not requested or not requested.issubset(APPROVED_TAGS):
            raise ValueError(
                f"tags must be a subset of the approved list {sorted(APPROVED_TAGS)}, got {v!r}"
            )
        return ",".join(sorted(requested))


class NucleiTool(ToolAdapter):
    """Nuclei scans, restricted to an explicitly approved, low-risk tag set.
    Exploit/intrusive template categories are never selectable through this
    adapter."""

    name = "nuclei"
    binary = "nuclei"

    def register_operations(self) -> None:
        self._op(OperationSpec(
            name="scan_approved_templates",
            description="Run nuclei with an approved, non-intrusive template tag set only.",
            args_model=NucleiScanArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: [
                "nuclei", "-u", a.url, "-tags", a.tags,
                "-silent", "-jsonl",
            ],
        ))

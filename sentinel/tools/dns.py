from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

from sentinel.schemas import ActionRisk
from sentinel.tools.base import OperationSpec, ToolAdapter

_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+$")
_RECORD_TYPES = {"A", "AAAA", "MX", "TXT", "NS", "CNAME", "SOA", "ANY"}


class DnsArgs(BaseModel):
    name: str
    record_type: str = "A"

    @field_validator("name")
    @classmethod
    def _vn(cls, v: str) -> str:
        v = v.strip()
        if not _HOST_RE.match(v):
            raise ValueError(f"invalid name: {v!r}")
        return v

    @field_validator("record_type")
    @classmethod
    def _vt(cls, v: str) -> str:
        v = v.strip().upper()
        if v not in _RECORD_TYPES:
            raise ValueError(f"unsupported record type: {v!r}")
        return v


class DnsTool(ToolAdapter):
    name = "dns"
    binary = "dig"

    def register_operations(self) -> None:
        self._op(OperationSpec(
            name="lookup",
            description="Resolve a DNS record (A/AAAA/MX/TXT/NS/CNAME/SOA) via dig, +short off "
                        "so the full answer section is captured for parsing.",
            args_model=DnsArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: ["dig", a.name, a.record_type],
        ))

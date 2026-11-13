from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

from sentinel.schemas import ActionRisk
from sentinel.tools.base import OperationSpec, ToolAdapter

_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+$")


class SmbArgs(BaseModel):
    host: str

    @field_validator("host")
    @classmethod
    def _vh(cls, v: str) -> str:
        v = v.strip()
        if not _HOST_RE.match(v):
            raise ValueError(f"invalid host: {v!r}")
        return v


class SmbTool(ToolAdapter):
    """SMB share/user enumeration using enum4linux-ng and an anonymous
    smbclient share listing. Read-only enumeration only."""

    name = "smb"
    binary = "enum4linux-ng"

    def register_operations(self) -> None:
        self._op(OperationSpec(
            name="enum_shares_users",
            description="Enumerate SMB shares, users, OS info, and policy via enum4linux-ng.",
            args_model=SmbArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: ["enum4linux-ng", "-A", a.host],
        ))
        self._op(OperationSpec(
            name="list_shares_anonymous",
            description="List shares via an anonymous smbclient session (no credentials).",
            args_model=SmbArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: ["smbclient", "-L", f"//{a.host}/", "-N"],
        ))

from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

from sentinel.schemas import ActionRisk
from sentinel.tools.base import OperationSpec, ToolAdapter

_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+$")


class FtpArgs(BaseModel):
    host: str
    port: int = 21

    @field_validator("host")
    @classmethod
    def _vh(cls, v: str) -> str:
        v = v.strip()
        if not _HOST_RE.match(v):
            raise ValueError(f"invalid host: {v!r}")
        return v

    @field_validator("port")
    @classmethod
    def _vp(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError("port out of range")
        return v


class FtpTool(ToolAdapter):
    """Passive FTP inspection: banner grab and well-known anonymous-login
    check via nmap NSE. Does not attempt credential brute forcing."""

    name = "ftp"
    binary = "nmap"

    def register_operations(self) -> None:
        self._op(OperationSpec(
            name="banner_and_anon_check",
            description="Grab FTP banner and check (read-only) whether anonymous login "
                        "is permitted, via nmap's ftp-anon script.",
            args_model=FtpArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: [
                "nmap", "-Pn", "-p", str(a.port),
                "--script", "ftp-anon,ftp-syst",
                "-oN", "-", a.host,
            ],
        ))

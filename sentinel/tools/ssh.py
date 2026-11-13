from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

from sentinel.schemas import ActionRisk
from sentinel.tools.base import OperationSpec, ToolAdapter

_HOST_RE = re.compile(r"^[A-Za-z0-9.\-]+$")


class BannerArgs(BaseModel):
    host: str
    port: int = 22

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


class SshTool(ToolAdapter):
    """Passive SSH service inspection only — banner grab and algorithm
    enumeration via nmap's ssh2-enum-algos script. No authentication
    attempts, no credential testing of any kind."""

    name = "ssh"
    binary = "nmap"

    def register_operations(self) -> None:
        self._op(OperationSpec(
            name="banner_grab",
            description="Grab the SSH banner and enumerate supported key exchange/host key "
                        "algorithms via nmap NSE (read-only, no auth attempted).",
            args_model=BannerArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: [
                "nmap", "-Pn", "-p", str(a.port),
                "--script", "ssh2-enum-algos,ssh-hostkey",
                "-oN", "-", a.host,
            ],
        ))

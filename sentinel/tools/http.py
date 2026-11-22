from __future__ import annotations

import re

from pydantic import BaseModel, field_validator

from sentinel.schemas import ActionRisk
from sentinel.tools.base import OperationSpec, ToolAdapter

_URL_RE = re.compile(r"^https?://[A-Za-z0-9.\-]+(:\d+)?(/.*)?$")
_WORDLIST_RE = re.compile(r"^[A-Za-z0-9._\-/]+$")


class FetchArgs(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def _v(cls, v: str) -> str:
        v = v.strip()
        if not _URL_RE.match(v):
            raise ValueError(f"invalid URL: {v!r}")
        return v


class FingerprintArgs(BaseModel):
    url: str

    @field_validator("url")
    @classmethod
    def _v(cls, v: str) -> str:
        v = v.strip()
        if not _URL_RE.match(v):
            raise ValueError(f"invalid URL: {v!r}")
        return v


class ContentDiscoveryArgs(BaseModel):
    url: str
    wordlist: str = "/usr/share/wordlists/dirb/common.txt"
    extensions: str | None = None  # "php,txt,html"

    @field_validator("url")
    @classmethod
    def _vu(cls, v: str) -> str:
        v = v.strip()
        if not _URL_RE.match(v):
            raise ValueError(f"invalid URL: {v!r}")
        return v

    @field_validator("wordlist")
    @classmethod
    def _vw(cls, v: str) -> str:
        if not _WORDLIST_RE.match(v):
            raise ValueError(f"invalid wordlist path: {v!r}")
        return v

    @field_validator("extensions")
    @classmethod
    def _ve(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.match(r"^[A-Za-z0-9,]+$", v):
            raise ValueError(f"invalid extensions spec: {v!r}")
        return v


class HttpTool(ToolAdapter):
    """HTTP/HTTPS inspection: headers via curl, fingerprinting via WhatWeb,
    content discovery via Gobuster — all read-only / non-destructive."""

    name = "http"
    binary = "curl"

    def register_operations(self) -> None:
        self._op(OperationSpec(
            name="fetch_headers",
            description="Fetch response headers and status line only (curl -I).",
            args_model=FetchArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: ["curl", "-s", "-S", "-I", "--max-time", "15", a.url],
        ))
        self._op(OperationSpec(
            name="fingerprint",
            description="Identify web technologies in use (WhatWeb).",
            args_model=FingerprintArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: ["whatweb", "--color=never", "-a", "1", a.url],
            binary="whatweb",
        ))
        self._op(OperationSpec(
            name="content_discovery",
            description="Directory/file discovery with a wordlist (Gobuster dir mode). "
                        "Non-destructive but generates many requests, so treated as needing approval.",
            args_model=ContentDiscoveryArgs,
            risk=ActionRisk.INTRUSIVE,
            build_argv=lambda a: (
                ["gobuster", "dir", "-u", a.url, "-w", a.wordlist, "-q", "-t", "20"]
                + (["-x", a.extensions] if a.extensions else [])
            ),
            binary="gobuster",
        ))

    def is_installed(self) -> bool:
        # The "http" logical tool wraps three binaries; report installed only
        # if the primary one (curl) is present. Status listing checks the rest
        # individually via the tool registry status command.
        return super().is_installed()

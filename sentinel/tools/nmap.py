from __future__ import annotations

import re
import subprocess
import xml.etree.ElementTree as ET
from typing import Any

from pydantic import BaseModel, field_validator

from sentinel.schemas import ActionRisk
from sentinel.tools.base import OperationSpec, ToolAdapter

_TARGET_RE = re.compile(r"^[A-Za-z0-9.\-:/]+$")


def _validate_target(v: str) -> str:
    v = v.strip()
    if not v or not _TARGET_RE.match(v) or v.startswith("-"):
        raise ValueError(f"invalid target: {v!r}")
    return v


class DiscoverArgs(BaseModel):
    target: str

    @field_validator("target")
    @classmethod
    def _v(cls, v: str) -> str:
        return _validate_target(v)


class ServiceScanArgs(BaseModel):
    target: str
    ports: str | None = None  # e.g. "22,80,443" — validated below

    @field_validator("target")
    @classmethod
    def _vt(cls, v: str) -> str:
        return _validate_target(v)

    @field_validator("ports")
    @classmethod
    def _vp(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not re.match(r"^[0-9,\-]+$", v):
            raise ValueError(f"invalid port spec: {v!r}")
        return v


def parse_nmap_xml(xml_text: str) -> dict[str, Any]:
    """Parse nmap -oX output into a plain dict of hosts/ports/services.
    Never regex-scrapes nmap's human-readable text output."""
    hosts: list[dict[str, Any]] = []
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError as exc:
        return {"hosts": [], "parse_error": str(exc)}

    for host_el in root.findall("host"):
        addr_el = host_el.find("address")
        addr = addr_el.get("addr") if addr_el is not None else None
        status_el = host_el.find("status")
        state = status_el.get("state") if status_el is not None else "unknown"
        ports: list[dict[str, Any]] = []
        ports_el = host_el.find("ports")
        if ports_el is not None:
            for port_el in ports_el.findall("port"):
                state_el = port_el.find("state")
                port_state = state_el.get("state") if state_el is not None else "unknown"
                service_el = port_el.find("service")
                ports.append({
                    "port": int(port_el.get("portid")),
                    "protocol": port_el.get("protocol"),
                    "state": port_state,
                    "service_name": service_el.get("name") if service_el is not None else None,
                    "product": service_el.get("product") if service_el is not None else None,
                    "version": service_el.get("version") if service_el is not None else None,
                    "extra_info": service_el.get("extrainfo") if service_el is not None else None,
                })
        hosts.append({"address": addr, "state": state, "ports": ports})
    return {"hosts": hosts}


class NmapTool(ToolAdapter):
    name = "nmap"
    binary = "nmap"

    def register_operations(self) -> None:
        self._op(OperationSpec(
            name="discover",
            description="Fast host/port discovery (top ports, no version detection).",
            args_model=DiscoverArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: ["nmap", "-Pn", "-T4", "--top-ports", "1000", "-oX", "-", a.target],
        ))
        self._op(OperationSpec(
            name="service_scan",
            description="Service/version detection against specific ports (or all discovered open ports).",
            args_model=ServiceScanArgs,
            risk=ActionRisk.SAFE,
            build_argv=lambda a: (
                ["nmap", "-Pn", "-sV", "-T4", "-oX", "-"]
                + (["-p", a.ports] if a.ports else ["-p-"])
                + [a.target]
            ),
        ))

    def version(self) -> str | None:
        try:
            out = subprocess.run(["nmap", "--version"], capture_output=True, text=True, timeout=5)
            first_line = out.stdout.splitlines()[0] if out.stdout else ""
            return first_line.strip() or None
        except Exception:
            return None

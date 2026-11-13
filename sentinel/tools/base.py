"""Tool adapter base class and registry.

Every security tool Sentinel can invoke is wrapped by a subclass of
`ToolAdapter`. Adapters never build shell strings from model output —
they expose a fixed set of named `operations`, each with a Pydantic
argument schema. The executor resolves (tool_name, operation, args)
against this registry and calls `build_argv`, which returns a literal
argv list handed straight to `subprocess`. No shell=True, anywhere.
"""
from __future__ import annotations

import abc
import shutil
from dataclasses import dataclass, field
from typing import Any, Callable, Type

from pydantic import BaseModel

from sentinel.schemas import ActionRisk


@dataclass
class OperationSpec:
    name: str
    description: str
    args_model: Type[BaseModel]
    risk: ActionRisk
    build_argv: Callable[[BaseModel], list[str]]
    parse_output: Callable[[str, str, int], dict[str, Any]] | None = None


@dataclass
class ToolStatus:
    tool_name: str
    binary: str
    installed: bool
    version: str | None
    tested_ok: bool
    operations: list[str] = field(default_factory=list)
    notes: str = ""


class ToolAdapter(abc.ABC):
    name: str
    binary: str  # executable looked up on PATH

    def __init__(self) -> None:
        self._operations: dict[str, OperationSpec] = {}
        self.register_operations()

    @abc.abstractmethod
    def register_operations(self) -> None:
        """Populate self._operations via self._op(...)."""

    def _op(self, spec: OperationSpec) -> None:
        self._operations[spec.name] = spec

    def operations(self) -> dict[str, OperationSpec]:
        return dict(self._operations)

    def get_operation(self, name: str) -> OperationSpec | None:
        return self._operations.get(name)

    def is_installed(self) -> bool:
        return shutil.which(self.binary) is not None

    def version(self) -> str | None:
        """Override for tools that support a fast --version check.
        Must not raise; return None if unavailable."""
        return None

    def status(self) -> ToolStatus:
        installed = self.is_installed()
        return ToolStatus(
            tool_name=self.name,
            binary=self.binary,
            installed=installed,
            version=self.version() if installed else None,
            tested_ok=False,  # set true only after an actual successful invocation
            operations=list(self._operations.keys()),
        )


class ToolRegistry:
    def __init__(self) -> None:
        self._tools: dict[str, ToolAdapter] = {}

    def register(self, adapter: ToolAdapter) -> None:
        self._tools[adapter.name] = adapter

    def get(self, name: str) -> ToolAdapter | None:
        return self._tools.get(name)

    def all(self) -> list[ToolAdapter]:
        return list(self._tools.values())

    def resolve(self, tool_name: str, operation: str) -> tuple[ToolAdapter, OperationSpec]:
        tool = self.get(tool_name)
        if tool is None:
            raise KeyError(f"Unknown tool '{tool_name}'")
        op = tool.get_operation(operation)
        if op is None:
            raise KeyError(f"Tool '{tool_name}' has no operation '{operation}'")
        return tool, op


def default_registry() -> ToolRegistry:
    """Build and return the registry with all built-in adapters registered."""
    from sentinel.tools.nmap import NmapTool
    from sentinel.tools.http import HttpTool
    from sentinel.tools.ssh import SshTool
    from sentinel.tools.ftp import FtpTool
    from sentinel.tools.smb import SmbTool
    from sentinel.tools.dns import DnsTool
    from sentinel.tools.nuclei import NucleiTool
    from sentinel.tools.zap import ZapTool

    registry = ToolRegistry()
    for cls in (NmapTool, HttpTool, SshTool, FtpTool, SmbTool, DnsTool, NucleiTool, ZapTool):
        registry.register(cls())
    return registry

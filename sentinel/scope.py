"""Authorized-target scope management.

This is the single source of truth for "am I allowed to touch this host".
Nothing in the planner or model layer can add to or bypass scope — only
this module, invoked from application code, can.
"""
from __future__ import annotations

import ipaddress
import re
from dataclasses import dataclass

from sentinel.schemas import ScopeEntry
from sentinel.state_store import StateStore

_HOSTNAME_RE = re.compile(
    r"^(?=.{1,253}$)(?!-)[A-Za-z0-9-]{1,63}(?<!-)(\.(?!-)[A-Za-z0-9-]{1,63}(?<!-))*$"
)


class ScopeError(Exception):
    pass


@dataclass
class ScopeCheckResult:
    allowed: bool
    scope_id: str | None
    reason: str


def _looks_like_valid_target(target: str) -> bool:
    target = target.strip()
    if not target:
        return False
    try:
        ipaddress.ip_network(target, strict=False)
        return True
    except ValueError:
        pass
    host = target.split("/")[0]
    return bool(_HOSTNAME_RE.match(host))


class ScopeManager:
    def __init__(self, store: StateStore):
        self.store = store

    def add(self, target: str, label: str = "", notes: str = "", confirmed: bool = False) -> ScopeEntry:
        target = target.strip()
        if not _looks_like_valid_target(target):
            raise ScopeError(f"'{target}' does not look like a valid host/IP/CIDR.")
        if not confirmed:
            raise ScopeError(
                "Target not added. You must explicitly confirm you are authorized "
                "to test this target (pass confirmation at the CLI prompt)."
            )
        entry = ScopeEntry(target=target, label=label, notes=notes, authorized_by_user=True)
        self.store.add_scope_entry(entry)
        return entry

    def remove(self, scope_id: str) -> bool:
        return self.store.remove_scope_entry(scope_id)

    def list(self) -> list[ScopeEntry]:
        return self.store.list_scope()

    def check(self, target: str) -> ScopeCheckResult:
        """Is `target` covered by an authorized scope entry?

        A target is in scope if it exactly matches an entry, or falls
        inside an entry defined as a CIDR network.
        """
        target = target.strip()
        for entry in self.list():
            if not entry.authorized_by_user:
                continue
            if entry.target == target:
                return ScopeCheckResult(True, entry.scope_id, "exact match")
            try:
                network = ipaddress.ip_network(entry.target, strict=False)
                candidate = ipaddress.ip_address(target)
                if candidate in network:
                    return ScopeCheckResult(True, entry.scope_id, f"within {entry.target}")
            except ValueError:
                continue
        return ScopeCheckResult(False, None, "no matching authorized scope entry")

    def require_in_scope(self, target: str) -> ScopeEntry:
        result = self.check(target)
        if not result.allowed:
            raise ScopeError(
                f"Target '{target}' is not in authorized scope. "
                f"Start with 'sentinel -a <target>' and confirm authorization."
            )
        entry = self.store.get_scope_entry(result.scope_id)
        assert entry is not None
        return entry

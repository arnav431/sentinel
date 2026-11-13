from __future__ import annotations

import pytest

from sentinel.scope import ScopeError, ScopeManager


def test_add_requires_confirmation(scope: ScopeManager):
    with pytest.raises(ScopeError):
        scope.add("10.10.10.5", confirmed=False)


def test_add_rejects_invalid_target(scope: ScopeManager):
    with pytest.raises(ScopeError):
        scope.add("; rm -rf /", confirmed=True)


def test_add_and_check_exact_match(scope: ScopeManager):
    scope.add("10.10.10.5", confirmed=True)
    result = scope.check("10.10.10.5")
    assert result.allowed is True


def test_check_rejects_out_of_scope(scope: ScopeManager):
    scope.add("10.10.10.5", confirmed=True)
    result = scope.check("10.10.10.6")
    assert result.allowed is False


def test_cidr_scope_covers_member_ip(scope: ScopeManager):
    scope.add("10.10.10.0/24", confirmed=True)
    result = scope.check("10.10.10.42")
    assert result.allowed is True


def test_require_in_scope_raises_for_unlisted_target(scope: ScopeManager):
    with pytest.raises(ScopeError):
        scope.require_in_scope("192.168.1.1")


def test_remove_scope_entry(scope: ScopeManager):
    entry = scope.add("10.10.10.5", confirmed=True)
    assert scope.remove(entry.scope_id) is True
    assert scope.check("10.10.10.5").allowed is False

from __future__ import annotations

import subprocess
from unittest.mock import patch

from sentinel.config import Settings
from sentinel.executor import Executor
from sentinel.policy import ValidatedAction
from sentinel.schemas import ActionRisk
from sentinel.state_store import StateStore
from sentinel.tools.nmap import DiscoverArgs
from sentinel.tools.base import default_registry


def _validated_echo_action():
    registry = default_registry()
    tool, op = registry.resolve("nmap", "discover")
    args = DiscoverArgs(target="127.0.0.1")
    return ValidatedAction(
        tool=tool, operation=op, args=args,
        argv=["echo", "hello"], risk=ActionRisk.SAFE, needs_approval=False,
    )


def test_executor_runs_and_persists_invocation(store: StateStore, tmp_settings: Settings):
    executor = Executor(store, tmp_settings)
    validated = _validated_echo_action()
    inv = executor.run("assess_test", validated)
    assert inv.exit_code == 0
    stdout, _ = executor.read_output(inv)
    assert "hello" in stdout

    persisted = store.list_invocations("assess_test")
    assert len(persisted) == 1
    assert persisted[0].invocation_id == inv.invocation_id


def test_executor_handles_timeout(store: StateStore, tmp_settings: Settings):
    executor = Executor(store, tmp_settings)
    validated = _validated_echo_action()
    validated.argv = ["sleep", "10"]
    tmp_settings.tool_timeout_s = 1
    inv = executor.run("assess_timeout", validated)
    assert inv.timed_out is True
    assert inv.error is not None


def test_executor_handles_missing_binary(store: StateStore, tmp_settings: Settings):
    executor = Executor(store, tmp_settings)
    validated = _validated_echo_action()
    validated.argv = ["this-binary-does-not-exist-xyz"]
    inv = executor.run("assess_missing", validated)
    assert inv.error is not None
    assert inv.exit_code is None


def test_executor_never_uses_shell_true(store: StateStore, tmp_settings: Settings):
    executor = Executor(store, tmp_settings)
    validated = _validated_echo_action()
    with patch("subprocess.run", wraps=subprocess.run) as mock_run:
        executor.run("assess_shell_check", validated)
        _, kwargs = mock_run.call_args
        assert kwargs.get("shell", False) is False


def test_executor_caps_output_size(store: StateStore, tmp_settings: Settings):
    executor = Executor(store, tmp_settings)
    tmp_settings.max_output_bytes = 10
    validated = _validated_echo_action()
    validated.argv = ["python3", "-c", "print('x' * 1000)"]
    inv = executor.run("assess_cap", validated)
    stdout, _ = executor.read_output(inv)
    assert len(stdout) <= 10

from __future__ import annotations

import os

import pytest
from typer.testing import CliRunner

runner = CliRunner()


@pytest.fixture(autouse=True)
def _isolated_data_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("SENTINEL_DATA_DIR", str(tmp_path / "data"))
    yield


def _app():
    # Imported lazily, after SENTINEL_DATA_DIR is patched for each test,
    # since sentinel.cli reads settings lazily inside each command (not at
    # import time) this is safe to import once per test module.
    from sentinel.cli import app
    return app


def test_tools_command_runs():
    result = runner.invoke(_app(), ["tools"])
    assert result.exit_code == 0
    assert "nmap" in result.stdout


def test_scope_add_without_confirmation_flag_prompts_and_can_be_declined():
    result = runner.invoke(_app(), ["scope", "add", "10.10.10.5"], input="n\n")
    assert result.exit_code != 0


def test_scope_add_with_yes_flag_succeeds():
    result = runner.invoke(_app(), ["scope", "add", "10.10.10.5", "--yes"])
    assert result.exit_code == 0
    assert "Added to scope" in result.stdout


def test_scope_add_rejects_malicious_target():
    result = runner.invoke(_app(), ["scope", "add", "$(whoami)", "--yes"])
    assert result.exit_code != 0


def test_scope_list_shows_added_target():
    runner.invoke(_app(), ["scope", "add", "10.10.10.5", "--yes"])
    result = runner.invoke(_app(), ["scope", "list"])
    assert "10.10.10.5" in result.stdout


def test_status_with_no_assessments_reports_cleanly():
    result = runner.invoke(_app(), ["status"])
    assert result.exit_code != 0
    assert "No assessments found" in result.stdout


def test_findings_with_no_assessments_reports_cleanly():
    result = runner.invoke(_app(), ["findings"])
    assert result.exit_code != 0


def test_scope_remove_unknown_id_reports_cleanly():
    result = runner.invoke(_app(), ["scope", "remove", "scope_doesnotexist"])
    assert result.exit_code == 0
    assert "No such scope entry" in result.stdout

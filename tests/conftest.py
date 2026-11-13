from __future__ import annotations

import os
from pathlib import Path

import pytest

from sentinel.config import Settings
from sentinel.scope import ScopeManager
from sentinel.state_store import StateStore
from sentinel.tools.base import default_registry


@pytest.fixture()
def tmp_settings(tmp_path: Path) -> Settings:
    data_dir = tmp_path / "data"
    return Settings(
        data_dir=data_dir,
        db_path=data_dir / "sentinel.db",
        evidence_dir=data_dir / "evidence",
        reports_dir=data_dir / "reports",
        tool_timeout_s=5,
    )


@pytest.fixture()
def store(tmp_settings: Settings) -> StateStore:
    return StateStore(tmp_settings.db_path)


@pytest.fixture()
def registry():
    return default_registry()


@pytest.fixture()
def scope(store: StateStore) -> ScopeManager:
    return ScopeManager(store)

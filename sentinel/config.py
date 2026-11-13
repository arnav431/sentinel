"""Central configuration, overridable via environment variables."""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def _bool(env_name: str, default: bool) -> bool:
    val = os.environ.get(env_name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_path(name: str) -> str | None:
    val = os.environ.get(name, "").strip()
    return val or None


@dataclass
class Settings:
    data_dir: Path
    db_path: Path
    evidence_dir: Path
    reports_dir: Path

    openrouter_api_key: str = field(default_factory=lambda: os.environ.get("OPENROUTER_API_KEY", ""))
    openrouter_base_url: str = field(default_factory=lambda: os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"))
    openrouter_model: str = field(default_factory=lambda: os.environ.get("OPENROUTER_MODEL", "nvidia/nemotron-3.5-lightning:free"))
    knowledge_dir: Path = field(default_factory=lambda: Path(os.environ.get("SENTINEL_KNOWLEDGE_DIR", "./knowledge")).resolve())
    knowledge_db_path: Path | None = field(default_factory=lambda: Path(os.environ["SENTINEL_KNOWLEDGE_DB"]).resolve() if os.environ.get("SENTINEL_KNOWLEDGE_DB") else None)
    model_request_timeout_s: int = field(default_factory=lambda: int(os.environ.get("SENTINEL_MODEL_TIMEOUT_S", "120")))

    tool_timeout_s: int = field(default_factory=lambda: int(os.environ.get("SENTINEL_TOOL_TIMEOUT_S", "180")))
    max_output_bytes: int = field(default_factory=lambda: int(os.environ.get("SENTINEL_MAX_OUTPUT_BYTES", str(5 * 1024 * 1024))))
    max_iterations_default: int = field(default_factory=lambda: int(os.environ.get("SENTINEL_MAX_ITERATIONS", "40")))
    max_concurrent_tasks: int = field(default_factory=lambda: int(os.environ.get("SENTINEL_MAX_CONCURRENT_TASKS", "1")))

    require_approval_for_intrusive: bool = field(default_factory=lambda: _bool("SENTINEL_REQUIRE_APPROVAL", True))
    allow_network_lookups: bool = field(default_factory=lambda: _bool("SENTINEL_ALLOW_NETWORK_LOOKUPS", False))


def get_settings() -> Settings:
    data_dir = Path(_env_path("SENTINEL_DATA_DIR") or "./data").resolve()
    db_path = Path(_env_path("SENTINEL_DB_PATH") or str(data_dir / "sentinel.db"))
    evidence_dir = Path(_env_path("SENTINEL_EVIDENCE_DIR") or str(data_dir / "evidence"))
    reports_dir = Path(_env_path("SENTINEL_REPORTS_DIR") or str(data_dir / "reports"))

    data_dir.mkdir(parents=True, exist_ok=True)
    evidence_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    return Settings(
        data_dir=data_dir,
        db_path=db_path,
        evidence_dir=evidence_dir,
        reports_dir=reports_dir,
    )

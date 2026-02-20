"""
config/settings.py
Centralised settings loaded from environment variables / .env file.
All tuneable parameters for the quantum engine live here.
"""

import logging
import logging.config
import yaml
from functools import lru_cache
from pathlib import Path

from typing import Optional

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── API ──────────────────────────────────────────────────────────────────
    api_host: str = "0.0.0.0"
    api_port: int = 8000
    api_workers: int = Field(default=1, description="Keep at 1 — GPU state is not fork-safe")

    # ── Logging ──────────────────────────────────────────────────────────────
    log_level: str = Field(default="INFO", pattern="^(DEBUG|INFO|WARNING|ERROR)$")

    # ── Quantum Backend ──────────────────────────────────────────────────────
    backend: str = Field(default="gpu", pattern="^(gpu|cpu)$")
    gpu_device: int = Field(default=0, ge=0)
    precision: str = Field(default="double", pattern="^(double|single)$")

    # ── Simulation Defaults (overridable per-request) ─────────────────────────
    num_shots: int = Field(default=8192, ge=128)
    max_qubits: int = Field(default=20, ge=4, le=30)

    # ── cuQuantum ────────────────────────────────────────────────────────────
    cuquantum_logfile: str = "/app/logs/cuquantum.log"

    # ── Databento ────────────────────────────────────────────────────────────
    databento_api_key: Optional[str] = Field(default=None, description="Databento Pay-As-You-Go API key")


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def setup_logging() -> None:
    """Load logging config from YAML. Falls back to basicConfig if file missing
    or if the log directory doesn't exist (e.g. outside Docker)."""
    log_yaml = Path(__file__).parent / "logging.yaml"
    configured = False
    if log_yaml.exists():
        with open(log_yaml) as f:
            cfg = yaml.safe_load(f)
        # Ensure log directories exist; skip silently if no write permission
        # (e.g. /app/logs outside Docker).
        for handler in cfg.get("handlers", {}).values():
            if "filename" in handler:
                log_path = Path(handler["filename"])
                try:
                    log_path.parent.mkdir(parents=True, exist_ok=True)
                except (PermissionError, OSError):
                    pass
        try:
            logging.config.dictConfig(cfg)
            configured = True
        except Exception:
            pass   # fall through to basicConfig

    if not configured:
        logging.basicConfig(
            level=get_settings().log_level,
            format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        )
    logging.getLogger(__name__).info("Logging initialised | level=%s", get_settings().log_level)

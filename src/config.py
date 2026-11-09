from pathlib import Path
from types import SimpleNamespace
from typing import Any

import yaml
from dotenv import load_dotenv

Config = SimpleNamespace


def _to_ns(value: Any) -> Any:
    if isinstance(value, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in value.items()})
    return value


def load_config(path: str = "config.yaml") -> Config:
    """Load YAML config into nested attributes; also loads .env (API keys stay in env vars)."""
    load_dotenv()
    with Path(path).open(encoding="utf-8") as f:
        return _to_ns(yaml.safe_load(f))

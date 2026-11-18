"""Load YAML configs from conf/."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_config(name: str = "data") -> dict[str, Any]:
    """Return conf/<name>.yaml as a dict."""
    with (REPO_ROOT / "conf" / f"{name}.yaml").open() as f:
        return yaml.safe_load(f)


def repo_path(p: str | Path) -> Path:
    """Resolve a config path relative to the repository root."""
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p

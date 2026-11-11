from __future__ import annotations

from pathlib import Path

import yaml


def load_declared_features(path: str | Path) -> list[str]:
    """Return feature names declared in the registry, rejecting duplicates."""
    with Path(path).open(encoding="utf-8") as handle:
        spec = yaml.safe_load(handle)
    names = [entry["name"] for entry in spec["features"]]
    if len(names) != len(set(names)):
        raise ValueError("duplicate feature names in registry")
    return names

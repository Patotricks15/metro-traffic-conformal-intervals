"""Fail if any neural-network or neural-forecasting dependency is imported or declared."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BLOCKED = (
    "torch",
    "tensorflow",
    "jax",
    "keras",
    "neuralforecast",
    "neuralprophet",
    "darts",
    "tsai",
)
IMPORT_RE = re.compile(r"^\s*(?:from|import)\s+([A-Za-z_]\w*)", re.MULTILINE)


def main() -> int:
    violations: list[str] = []
    for path in sorted((ROOT / "src").rglob("*.py")) + sorted((ROOT / "tests").rglob("*.py")):
        text = path.read_text(encoding="utf-8")
        for match in IMPORT_RE.finditer(text):
            if match.group(1) in BLOCKED:
                violations.append(f"{path.relative_to(ROOT)}: import {match.group(1)}")

    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    for name in BLOCKED:
        if re.search(rf"[\"'\s]{re.escape(name)}[\s\"'<>=~!\[]", pyproject):
            violations.append(f"pyproject.toml: declares {name}")

    if violations:
        print("Blocked neural dependencies found:")
        print("\n".join(f"  - {v}" for v in violations))
        return 1
    print("No neural dependencies found.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Run the lightweight numerical tests without requiring pytest."""

from __future__ import annotations

import runpy
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    failures: list[str] = []
    completed = 0
    for path in sorted((root / "tests").glob("test_*.py")):
        namespace = runpy.run_path(str(path))
        for name, function in sorted(namespace.items()):
            if name.startswith("test_") and callable(function):
                label = f"{path.name}::{name}"
                try:
                    function()
                except Exception as error:  # noqa: BLE001 - test harness
                    failures.append(f"{label}: {type(error).__name__}: {error}")
                    print(f"FAIL {label}")
                else:
                    completed += 1
                    print(f"PASS {label}")
    if failures:
        print("\nFailures:")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print(f"\n{completed} core tests passed")


if __name__ == "__main__":
    main()


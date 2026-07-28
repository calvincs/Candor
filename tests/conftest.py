"""Test wiring only. No assertions live here.

`candor_conformance.py` ships a `pytest_configure` that registers the stage
markers, but pytest only calls that hook from conftest/plugins — never from a
test module. Registering them here keeps the harness file verbatim apart from
its documented `make_driver()` wiring (DEVIATIONS.md D1, D2).
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "src"
for path in (SRC, ROOT):          # ROOT so the §6.8 bench harness is importable
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))


def pytest_configure(config):
    for m in ("stage1", "stage2", "stage3", "stage4", "stage5",
              "fail_stop", "alert_only"):
        config.addinivalue_line("markers", f"{m}: CANDOR conformance tag (spec §6)")

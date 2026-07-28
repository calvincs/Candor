"""Shared fixtures for the additive unit suite."""

from __future__ import annotations

import pytest

from candor.system import CandorSystem


@pytest.fixture()
def sys_(tmp_path):
    system = CandorSystem(tmp_path / "store")
    yield system
    system.close()


@pytest.fixture()
def seeded(sys_):
    """The §8 seed path: registry + facts in through the gate, human as proposer."""
    for stmt in ({"pred": "reachable", "args": ["a", "b"], "stmt_type": "crisp"},
                 {"pred": "reachable", "args": ["b", "c"], "stmt_type": "crisp"},
                 {"pred": "flaky_link", "args": ["c", "d"], "stmt_type": "frequency"}):
        sys_.assert_(stmt, source="seed", actor="human:calvin")
    sys_.run_gate()
    return sys_

"""H2: reliability is a first-class ledger event, not a side file.

Two invariants the old `<root>/reliability_overrides.json` side file broke:

  I1  a ledger-only rebuild (derived index AND any side file deleted) must
      reproduce every committed number. Persisting an operator override outside
      the chain let `verify_chain()` stay true while the numbers silently moved.

  I3  an override composed with a settlement must give identical numbers live
      vs. after `replay()`. Re-applying the side file *after* the fold put the
      override at the wrong ledger position, so it diverged from the live path
      whenever a settlement touched the same actor.
"""

from __future__ import annotations

from candor.system import CandorSystem


def _build_crisp(root):
    m = CandorSystem(root)
    for a in ("tool:probe", "tool:bad", "human:me"):
        m.set_actor_quota(a, obs_per_epoch=10_000, cand_per_epoch=10_000)
    m.assert_({"pred": "holds", "args": ["t0"], "stmt_type": "crisp"},
              source="s", actor="human:me")
    m.run_gate()
    stmt = {"pred": "holds", "args": ["t0"]}
    for _ in range(10):
        m.observe(stmt, True, {}, actor="tool:probe")
        m.observe(stmt, False, {}, actor="tool:bad")
    m.run_gate()
    return m, stmt


def _p(m, stmt):
    return repr(m.predict(stmt, budget=1500).p)


def test_set_reliability_writes_no_side_file(tmp_path):
    root = tmp_path / "store"
    m, _ = _build_crisp(root)
    m.set_reliability("tool:bad", "external", 0.001, 100.0)
    exists = (root / "reliability_overrides.json").exists()
    m.close()
    assert not exists, "set_reliability persisted state outside the ledger"


def test_ledger_alone_reproduces_a_reliability_override(tmp_path):
    root = tmp_path / "store"
    m, stmt = _build_crisp(root)
    p_no_override = _p(m, stmt)
    m.set_reliability("tool:bad", "external", 0.001, 100.0)
    p_before = _p(m, stmt)
    assert p_before != p_no_override, "override did not move the belief (vacuous)"
    closure_before = m.closure_hash()
    m.close()

    # A ledger-only rebuild: drop the derived index AND any reliability side
    # file. The chain alone must carry the override (I1).
    for suffix in ("", "-wal", "-shm"):
        f = root / ("index.sqlite3" + suffix)
        if f.exists():
            f.unlink()
    side = root / "reliability_overrides.json"
    if side.exists():
        side.unlink()

    m2 = CandorSystem(root)
    p_after = _p(m2, stmt)
    closure_after = m2.closure_hash()
    side_after = (root / "reliability_overrides.json").exists()
    m2.close()

    assert p_after == p_before, "override lost on a ledger-only rebuild (I1)"
    assert closure_after == closure_before, "closure hash changed on rebuild (I1)"
    assert not side_after, "rebuild recreated a side file outside the ledger"


def test_override_then_settlement_is_replay_stable(tmp_path):
    m = CandorSystem(tmp_path / "store")
    m.set_actor_quota("tool:probe", obs_per_epoch=10_000, cand_per_epoch=10_000)
    m.assert_({"pred": "flaky_link", "args": ["c", "d"], "stmt_type": "frequency"},
              source="s", actor="human:me")
    m.run_gate()
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    m.observe(stmt, True, {}, actor="tool:probe")
    # Override BEFORE the settlement...
    m.set_reliability("tool:probe", "external", 2.0, 8.0)
    # ...then a deterministic_total settlement that scores the same actor.
    m.register_oracle("verifier:t", "deterministic_total", "t", "h", "e")
    cid = m.claim(stmt, "external", "verifier:t", due=0)
    m.resolve(cid, outcome=True)
    live = m.closure_hash()
    replayed = m.replay()
    m.close()
    assert replayed == live, (
        "override + settlement diverge live vs replay: the override folds at "
        "the wrong ledger position (I3)")

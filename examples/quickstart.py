"""The complete CANDOR loop in one file: admit, observe, predict, settle, audit.

Runs offline in a temp directory. python examples/quickstart.py
"""

import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from candor.system import CandorSystem  # noqa: E402

m = CandorSystem(Path(tempfile.mkdtemp(prefix="candor-demo-")))

# ── 1. everything enters as a candidate; the gate decides ───────────────────
m.assert_({"kind": "symbol", "pred": "boils_at", "arity": 2,
           "arg_types": ["substance", "temperature"],
           "canonical_units": {"1": "K"}}, source="seed", actor="human:me")
m.assert_({"pred": "boils_at", "args": ["water", "212F"], "stmt_type": "crisp"},
          source="doc:phys", actor="agent:reader")
m.assert_({"pred": "deploy_ok", "args": ["api"], "stmt_type": "frequency"},
          source="runbook", actor="human:me")
for run in m.run_gate():
    print(f"gate: {run['candidate_kind']:<8} -> {run['status']}")

# units were canonicalized at admission: 212F is stored as 373.15K and the
# same fact answers under either notation
fid = m.fact_id_for({"pred": "boils_at", "args": ["water", "100C"]})
print(f"boils_at(water, 100C) resolves to the same fact: {fid is not None}")

# ── 2. attributed observations with wide context ────────────────────────────
for i, ok in enumerate([True, True, False, True, True, False, True, True]):
    m.observe({"pred": "deploy_ok", "args": ["api"]}, ok,
              ctx={"day": "fri" if i % 3 == 2 else "weekday"}, actor="tool:ci")

# ── 3. a real probability, with an interval and a reproducible snapshot ─────
p = m.predict({"pred": "deploy_ok", "args": ["api"]}, budget=10_000)
print(f"p(deploy_ok) = {p.p:.3f}   ci = ({p.ci[0]:.2f}, {p.ci[1]:.2f})")

# ── 4. claims settle against an oracle; that is the ONLY thing that moves trust
m.register_oracle("verifier:ci", "deterministic_total", "ci", "h", "e")
cid = m.claim({"pred": "deploy_ok", "args": ["api"]}, frame="external",
              criterion="verifier:ci", due=0)
m.resolve(cid, outcome=True)

# ── 5. audit anything back to its events ────────────────────────────────────
why = m.why(m.fact_id_for({"pred": "deploy_ok", "args": ["api"]}))
print(f"raw counts by (actor|channel): {why['raw_counts']}")
print(f"context diversity: {why['source_diversity']}")

# ── 6. the index is disposable; the ledger is the truth ─────────────────────
before = m.closure_hash()
m.corrupt("drop_index")                 # test hook: delete SQLite entirely
assert m.replay() == before
print("dropped the index, replayed the ledger: closure hash identical")
m.close()

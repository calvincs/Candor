"""Drift with dates: conditions vs regime changes vs honest confusion.

Three streams that look identical to a decaying average, told apart by the
curiosity sweep:  a scraper whose success DEPENDS on method (-> guard), a tool
that BROKE on a specific day (-> located regime change), and variance with no
recorded explanation (-> an open question naming what to measure).
python examples/regime_change.py
"""

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from candor.system import CandorSystem  # noqa: E402

m = CandorSystem(Path(tempfile.mkdtemp(prefix="candor-drift-")))
rng = random.Random(11)

for pred in ("scrape_ok", "convert_ok", "flaky_ok"):
    m.assert_({"pred": pred, "args": ["svc"], "stmt_type": "frequency"},
              source="ops", actor="human:me")
m.run_gate()

# 1) success depends on a recorded covariate -> should become a GUARD
for i in range(60):
    method = "crawl4ai" if i % 2 else "http"
    ok = rng.random() < (0.92 if method == "crawl4ai" else 0.25)
    m.observe({"pred": "scrape_ok", "args": ["svc"]}, ok,
              {"method": method}, actor="tool:probe")

# 2) a one-way break at observation 30 -> should become a LOCATED regime change
for i in range(60):
    ok = rng.random() < (0.9 if i < 30 else 0.05)
    m.observe({"pred": "convert_ok", "args": ["svc"]}, ok,
              {"host": "box-a"}, actor="tool:probe")

# 3) variance driven by something never recorded -> an OPEN QUESTION
for i in range(240):
    hidden_block = (i // 20) % 2                 # the unlogged culprit
    ok = rng.random() < (0.85 if hidden_block else 0.45)
    m.observe({"pred": "flaky_ok", "args": ["svc"]}, ok,
              {"batch": str(i // 10)}, actor="tool:probe")

runs = m.run_gate()
print("what the sweep proposed and the gate ruled:")
for r in runs:
    print(f"  {r['candidate_kind']:<22} {r['status']}")

for q in m.questions():
    fact = m.index.one("SELECT pred FROM facts WHERE id=?", (q["target_id"],))
    print(f"\nopen question on {fact['pred']}:")
    print(f"  detected structure: {q['residual_partition']}")
    print(f"  suggested measurement: {q['suggested_measurement']}")

print("\nThe average for all three streams is ~0.5. The difference between "
      "'needs a condition', 'broke on a date', and 'we don't know yet' is the "
      "whole point.")
m.close()

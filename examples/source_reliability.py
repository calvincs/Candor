"""Trust that is earned and asymmetric: the two-coin model in action.

Three sources vote on claims. One is careful, one is noisy, one always says
yes. After a training slice settles against ground truth, the substrate has
learned each source's *shape* — and a fresh claim endorsed only by the
sycophant moves the posterior far less than one endorsed by the careful
source.  python examples/source_reliability.py
"""

import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from candor.core.committed import reliability as R  # noqa: E402
from candor.system import CandorSystem  # noqa: E402

m = CandorSystem(Path(tempfile.mkdtemp(prefix="candor-trust-")))
m.set_actor_quota("human:me", cand_per_epoch=2000)
m.register_oracle("verifier:truth", "deterministic_total", "gt", "h", "e")
rng = random.Random(7)

# ── 100 training claims with known outcomes; three voters ───────────────────
for i in range(100):
    stmt = {"pred": "claim_true", "args": [f"train{i}"], "stmt_type": "crisp"}
    m.assert_(stmt, source="suite", actor="human:me")
m.run_gate()
for i in range(100):
    truth = rng.random() < 0.5
    stmt = {"pred": "claim_true", "args": [f"train{i}"]}
    m.observe(stmt, truth if rng.random() < 0.95 else not truth, {},
              actor="agent:careful")                       # ~95% right
    m.observe(stmt, truth if rng.random() < 0.65 else not truth, {},
              actor="agent:noisy")                         # ~65% right
    m.observe(stmt, True, {}, actor="agent:sycophant")     # always yes
    cid = m.claim(stmt, frame="external", criterion="verifier:truth", due=0)
    m.resolve(cid, outcome=truth)                          # trust moves HERE

# ── what did it learn? ──────────────────────────────────────────────────────
print(f"{'source':<18} {'sens':>6} {'fpr':>6}   meaning")
for actor in ("agent:careful", "agent:noisy", "agent:sycophant"):
    sens, fpr = R.rates(R.confusion(m.index, actor))
    yes_lr = sens / fpr
    print(f"{actor:<18} {sens:>6.2f} {fpr:>6.2f}   a 'yes' multiplies the "
          f"odds by {yes_lr:.1f}x")

# ── the payoff: identical votes, very different evidence ────────────────────
for name, actor in (("endorsed by SYCOPHANT", "agent:sycophant"),
                    ("endorsed by CAREFUL", "agent:careful")):
    stmt = {"pred": "claim_true", "args": [f"fresh-{actor}"], "stmt_type": "crisp"}
    m.assert_(stmt, source="suite", actor="human:me")
    m.run_gate()
    m.observe(stmt, True, {}, actor=actor)
    p = m.predict(stmt, budget=10_000)
    print(f"fresh claim {name:<24} p = {p.p:.3f}")

print("\nSame vote, different voter, different posterior — and every number "
      "above is recomputable from integer counts in the ledger.")
m.close()

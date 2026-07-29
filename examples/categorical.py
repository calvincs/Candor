"""Open-vocabulary categorical facts, and a read-time X-ray of a flaky one.

Two capabilities the crisp/frequency path never had:

  1. A CATEGORICAL fact records a VALUE per observation, not true/false. The
     vocabulary GROWS as new values arrive, and "a value we have never seen" is
     a first-class outcome carrying real predictive mass — not an error bar.
  2. distribution() is a pure read-time breakdown of a flaky BINARY fact: it
     writes nothing, moves no count, changes no closure_hash — it just says HOW
     the fact splits by context and how much of the spread nothing recorded
     explains.

Runs offline in a temp directory and leaves no ./store behind.
python examples/categorical.py
"""

import shutil
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from candor.system import CandorSystem, CategoricalPrediction  # noqa: E402

store = Path(tempfile.mkdtemp(prefix="candor-cat-"))
m = CandorSystem(store)
m.set_actor_quota("tool:probe", obs_per_epoch=100_000, cand_per_epoch=100_000)


def show(p: CategoricalPrediction, title: str) -> None:
    print(f"\n{title}  (N={p.total_observations})")
    for value, sl in p.values.items():
        print(f"    {value:<14} p={sl.p:5.3f}  ci=({sl.ci[0]:.2f}, {sl.ci[1]:.2f})")
    u = p.unknown
    print(f"    {'<unseen>':<14} p={u.p:5.3f}  ci=({u.ci[0]:.2f}, {u.ci[1]:.2f})")


# ── Part 1 — an open vocabulary with a first-class unknown mass ──────────────
m.assert_({"pred": "resolves", "args": ["login"], "stmt_type": "categorical"},
          source="seed", actor="tool:probe")
m.run_gate()

for v in ["captcha"] * 8 + ["block"] * 2:            # a VALUE per obs, not T/F
    m.observe({"pred": "resolves", "args": ["login"]}, ctx={},
              actor="tool:probe", value=v)

p = m.predict({"pred": "resolves", "args": ["login"]}, budget=1000)
show(p, "resolves(login): what happens on a fresh login attempt?")
print(f"    snapshot: {p.snapshot_id}")
print("  -> captcha ~73%, block ~18%, and ~9% genuinely-unseen mass. A value we")
print("     have not observed yet is a first-class outcome, not an error bar.")

# the vocabulary GROWS: a brand-new value simply appears with its own slice
m.observe({"pred": "resolves", "args": ["login"]}, ctx={},
          actor="tool:probe", value="mfa_challenge")
p = m.predict({"pred": "resolves", "args": ["login"]}, budget=1000)
show(p, "one 'mfa_challenge' later — the vocabulary grew, no schema change:")

# ── Part 1b — a value CONDITIONED on context, if the sweep admits a guard ────
m.assert_({"pred": "renders", "args": ["home"], "stmt_type": "categorical"},
          source="seed", actor="tool:probe")
m.run_gate()
for region, value, times in (("eu", "captcha", 18), ("eu", "full_page", 2),
                             ("us", "full_page", 18), ("us", "captcha", 2)):
    for _ in range(times):
        m.observe({"pred": "renders", "args": ["home"]}, ctx={"region": region},
                  actor="tool:probe", value=value)
m.run_gate()                       # let the curiosity sweep discover a region guard
p = m.predict({"pred": "renders", "args": ["home"]}, budget=1000)
if p.by_context:
    print("\nrenders(home) conditioned on the region the sweep discovered:")
    for key, groups in p.by_context.items():
        for cval, grp in groups.items():
            top = max(grp["values"].items(), key=lambda kv: kv[1].p, default=None)
            if top is None:
                continue
            name, sl = top
            print(f"    {key}={cval:<12} (n={grp['n']:>2})  most-likely "
                  f"{name} p={sl.p:.3f}")
else:
    print("\n(no region guard admitted at these thresholds — by_context is empty,")
    print(" which is fine: the marginal distribution stands on its own.)")

# ── Part 2 — distribution(): a read-time X-ray of a flaky BINARY fact ────────
m.assert_({"pred": "flaky_api", "args": ["payments"], "stmt_type": "frequency"},
          source="ops", actor="tool:probe")
m.run_gate()
for ok in [True] * 27 + [False] * 3:                 # us-east: usually works
    m.observe({"pred": "flaky_api", "args": ["payments"]}, ok,
              ctx={"region": "us-east"}, actor="tool:probe")
for ok in [True] * 8 + [False] * 22:                 # eu: usually fails
    m.observe({"pred": "flaky_api", "args": ["payments"]}, ok,
              ctx={"region": "eu"}, actor="tool:probe")
m.run_gate()

d = m.distribution({"pred": "flaky_api", "args": ["payments"]})
print(f"\nflaky_api(payments): {d['n_obs']} obs, flaky={d['flaky']} "
      f"(dispersion_flag={d['dispersion_flag']})")
for key, buckets in d["modes"].items():
    print(f"  split by {key}:")
    for cval, cell in sorted(buckets.items()):
        print(f"    {cval:<12} p(True)={cell['p']:.2f}  (n={cell['n']})")
res = d["residual"]
print(f"  guard variable = {res['conditioning_key']}   "
      f"explained η²={res['explained']:.2f}   "
      f"unexplained={res['unexplained']:.2f}")

m.close()
shutil.rmtree(store, ignore_errors=True)
print("\nAn unseen value is a category, not a rounding error — and a flaky fact "
      "will tell you\nhow much of its own mess nothing on record can yet explain.")

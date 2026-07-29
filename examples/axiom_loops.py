"""The v0.6 axiom loops: postulate -> validate -> implement, four ways.

Four small worlds, one per Δ:
  Δ10  a fact whose variance is driven by TIME OF DAY, with nothing logged —
       the sweep postulates the frame itself (derived:hour) and gates it;
  Δ11  a condition that stops being true — the prospective audit demotes it
       through the ledger, and the memoryless sweep cannot flap it back in;
  Δ12  an analogy committed as a claim, settled true, and only THEN admitted
       as a fact — through the gate, like everything else;
  Δ13  a metric that decouples when it becomes the target — found BY NAME
       because the intervention was logged as a do: key, and the pooled
       prediction says regime_mixed out loud.
python examples/axiom_loops.py
"""

import json
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from candor.system import CandorSystem  # noqa: E402

HOUR, DAY = 3_600_000, 86_400_000
T0 = 1_749_945_600_000                  # a midnight, so hours read as planted
m = CandorSystem(Path(tempfile.mkdtemp(prefix="candor-axiom-")))
rng = random.Random(7)


def admitted_guards(pred):
    fid = m.fact_id_for({"pred": pred, "args": ["svc"]})
    out = []
    for r in m.index.query("SELECT body_json, status FROM candidates "
                           "WHERE kind='guard' ORDER BY event_seq"):
        b = json.loads(r["body_json"])
        if b.get("target_fact") == fid:
            out.append((r["status"], b["conditioning_key"],
                        b["body"]["guards"][0]["value"]))
    return out


# ── Δ10: the sweep postulates a frame nobody logged ─────────────────────────
m.assert_({"pred": "backup_ok", "args": ["svc"], "stmt_type": "frequency"},
          source="ops", actor="human:me")
m.run_gate()
for i in range(160):                    # 03:00 fails, 14:00 works; ctx is EMPTY
    hour = 3 if i % 2 == 0 else 14
    ok = rng.random() < (0.15 if hour == 3 else 0.9)
    m.observe({"pred": "backup_ok", "args": ["svc"]}, ok, {},
              actor="tool:probe", ts=T0 + (i // 2) * DAY + hour * HOUR)
m.run_gate()
print("Δ10 — nothing was logged, the frame was synthesized:")
for status, key, val in admitted_guards("backup_ok"):
    print(f"  guard {status}: {key} == {val!r}")
d = m.distribution({"pred": "backup_ok", "args": ["svc"]})
by_hour = d["derived_modes"]["derived:hour"]
print(f"  derived:hour breakdown: 03 -> {by_hour['03']['p']:.2f}, "
      f"14 -> {by_hour['14']['p']:.2f}  (η² explained: "
      f"{d['residual']['explained']:.2f})")

# ── Δ11: an admitted condition must keep paying rent ────────────────────────
m.assert_({"pred": "fetch_ok", "args": ["svc"], "stmt_type": "frequency"},
          source="ops", actor="human:me")
m.run_gate()


def feed_fetch(n, p_of, t0):
    for i in range(n):
        method = "crawl4ai" if i % 2 == 0 else "http"
        m.observe({"pred": "fetch_ok", "args": ["svc"]},
                  rng.random() < p_of(method), {"method": method},
                  actor="tool:probe", ts=T0 + 200 * DAY + (t0 + i) * 1000)


feed_fetch(200, lambda meth: 0.9 if meth == "crawl4ai" else 0.2, 0)
m.run_gate()                            # guard admitted on real structure
feed_fetch(200, lambda meth: 0.15 if meth == "crawl4ai" else 0.9, 200)
runs = m.run_gate()                     # ...then the world flips
print("\nΔ11 — the condition reversed, so the audit demoted it:")
for r in runs:
    if r["status"] == "demoted":
        print(f"  {r['reason']}")
print("  candidate rows now:", [s for s, _, _ in admitted_guards("fetch_ok")]
      or "(none admitted)")

# ── Δ12: an analogy becomes a fact only after it survives settlement ────────
for pred in ("flies", "glides"):
    m.assert_({"kind": "symbol", "pred": pred, "arity": 1, "arg_types": ["any"]},
              source="seed", actor="human:me")
m.run_gate()
for pred, arg in (("flies", "hawk"), ("flies", "eagle"),
                  ("glides", "hawk"), ("glides", "squirrel")):
    m.assert_({"pred": pred, "args": [arg], "stmt_type": "crisp"},
              source="doc", actor="agent:x")
m.assert_({"kind": "rule", "head": {"pred": "airborne", "args": ["?a"]},
           "body": {"literals": [{"pred": "flies", "args": ["?a"]},
                                 {"pred": "glides", "args": ["?a"]}]},
           "holdout": {"hits": 9, "misses": 1}},
          source="seed", actor="human:me")
m.run_gate()
goal = {"pred": "flies", "args": ["squirrel"]}
out = m.conjecture(goal, 0.15, commit=True)
c = out[0]
row = m.index.one("SELECT predicted_p FROM claims WHERE id=?", (c["claim_id"],))
print(f"\nΔ12 — conjectured {goal['pred']}(squirrel) via "
      f"{c['via']['pred']} (sim {c['sim']:.2f}); claim filed at "
      f"p={row['predicted_p']:.2f} (the ANALOG's earned probability)")
m.resolve(c["claim_id"], outcome=True)  # a field observer confirms it
m.run_gate()
print(f"  settled true -> admitted through the gate: "
      f"fact_id = {m.fact_id_for(goal)}")

# ── Δ13: acting on the world is not watching it ─────────────────────────────
m.assert_({"pred": "metric_tracks_goal", "args": ["svc"],
           "stmt_type": "frequency"}, source="ops", actor="human:me")
m.run_gate()
for i in range(200):                    # the metric becomes the target at 100
    targeted = i >= 100
    ok = rng.random() < (0.25 if targeted else 0.9)
    m.observe({"pred": "metric_tracks_goal", "args": ["svc"]}, ok,
              {"do:optimize_metric": "yes" if targeted else "no"},
              actor="tool:probe", ts=T0 + 400 * DAY + i * 1000)
m.run_gate()
print("\nΔ13 — Goodhart, found by name because the intervention was logged:")
for status, key, val in admitted_guards("metric_tracks_goal"):
    print(f"  guard {status}: {key} == {val!r}  (regime-dependent)")
p = m.predict({"pred": "metric_tracks_goal", "args": ["svc"]}, budget=10_000)
print(f"  pooled p={p.p:.2f} with caveats {sorted(p.caveats)} — the marginal "
      f"averages across an intervention boundary and says so")

print("\nEvery loop above ran postulate -> validate -> implement, and every "
      "structural change went through the same gate and the same ledger.")
m.close()

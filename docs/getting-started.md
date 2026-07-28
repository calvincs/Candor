# Getting started

## Install

Python 3.11+. The core is standard-library only; `pytest` and `hypothesis`
are needed only to run the test suite.

```sh
make venv          # creates .venv and installs the test deps
make gates         # stage1..stage5 conformance gates — should all be green
make unit          # 180 additive unit tests
```

To use CANDOR from your own code, put `src/` on your path (or install it as a
package) — there is nothing to compile and nothing to download.

## The mental model in five sentences

1. A **fact** is a statement like `fetch_ok(rrstar.com)` — either *crisp*
   ("this is true or it isn't") or *frequency* ("this succeeds at some rate").
2. Nothing becomes a fact by assertion alone: assertions become **candidates**,
   and a seven-step **gate** (schema check, unit canonicalization, sandbox,
   pin veto, held-out check, MDL, contradiction check) decides admission.
3. **Observations** are attributed outcome reports — *who* saw *what*, under
   *which context* — appended to a hash-chained ledger, never edited.
4. **Predictions** compose the stored evidence into a calibrated probability;
   when a prediction later **settles** against reality, every source that
   voted gets scored, which is the only way trust ever moves.
5. The **curiosity engine** watches for facts that behave inconsistently and
   proposes repairs: a *condition* ("true when pressure is low") or a *regime
   change* ("this stopped being true on April 30").

## A complete loop

```python
import sys; sys.path.insert(0, "src")
from candor.system import CandorSystem

m = CandorSystem("./store")

# --- 1. admit a fact (the seed path: human proposer, same gate as everyone)
m.assert_({"pred": "deploy_ok", "args": ["api"], "stmt_type": "frequency"},
          source="runbook", actor="human:me")
m.run_gate()

# --- 2. observe outcomes, wide context, attributed actors
for i, ok in enumerate([True, True, False, True]):
    m.observe({"pred": "deploy_ok", "args": ["api"]}, ok,
              ctx={"day": "mon" if i % 2 else "fri"}, actor="tool:ci")

# --- 3. predict: a real probability with an interval and a snapshot id
p = m.predict({"pred": "deploy_ok", "args": ["api"]}, budget=10_000)
print(f"p={p.p:.2f}  ci={p.ci}")

# --- 4. claim + settle: this is where sources earn or lose trust
m.register_oracle("verifier:ci", "deterministic_total", "ci-pipeline", "h", "e")
cid = m.claim({"pred": "deploy_ok", "args": ["api"]}, frame="external",
              criterion="verifier:ci", due=0)
m.resolve(cid, outcome=True)

# --- 5. audit anything
print(m.why(m.fact_id_for({"pred": "deploy_ok", "args": ["api"]})))

m.close()
```

Run `examples/quickstart.py` for this loop end to end, and
`examples/source_reliability.py` / `examples/regime_change.py` for the two
signature behaviours.

## Things that will surprise you (on purpose)

- **`assert_` never creates a fact.** It creates a candidate. If the gate
  rejects it, the rejection is recorded with the failing step — rejections are
  training signal, not garbage.
- **Retrieval cannot move a number.** `recall()` is logged to a side stream
  with no write path to any count — enforced by the import graph and audited
  mechanically (`make audit`).
- **Observations alone never change trust.** A thousand agreeable
  observations from a source leave its reliability untouched; only settled
  predictions score it. The world is allowed to surprise you; sources are not
  allowed to promote themselves.
- **Deleting `index.sqlite3` loses nothing.** It is a derived view; `replay()`
  rebuilds it bit-for-bit from the ledger segments.
- **Quotas are on by default.** An actor that floods observations or spams the
  gate hits its per-epoch quota and gets a `QuotaExceeded` — provision with
  `set_actor_quota()` for legitimate bulk loads.

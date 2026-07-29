# Getting started

## Install

Python 3.11+. The core is standard-library only; `pytest` and `hypothesis`
are needed only to run the test suite.

```sh
make venv          # creates .venv and installs the test deps
make gates         # stage1..stage5 conformance gates — should all be green
make unit          # 474 additive unit tests
make claims        # 41 claims tests: the README, measured on synthetic worlds
```

`make claims-fast` skips the prediction-heavy tests (~45s instead of ~2m30s),
and `CLAIMS_SCALE=4 make claims` raises the replication count when you want to
investigate a threshold rather than just check it.

To use CANDOR from your own code, put `src/` on your path (or install it as a
package) — there is nothing to compile and nothing to download.

## The mental model in five sentences

1. A **fact** is a statement like `fetch_ok(rrstar.com)` — *crisp* ("this is
   true or it isn't"), *frequency* ("this succeeds at some rate"), or
   *categorical* ("which of an open-ended set of values happened", with a
   first-class *unknown* mass for values not yet seen).
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
   change* ("this stopped being true on April 30") — and it searches frames
   you never logged (hour-of-day, the fact's own previous outcome, key
   interactions) alongside the context you did. Admitted conditions are not
   tenured: they keep being scored on later observations, and one that stops
   earning its keep is demoted through the same gate that admitted it.

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

Run `examples/quickstart.py` for this loop end to end,
`examples/source_reliability.py` / `examples/regime_change.py` for the two
signature behaviours, `examples/categorical.py` for open-vocabulary
categorical facts with a first-class unknown mass, and
`examples/axiom_loops.py` for the v0.6 loops: a frame the sweep synthesized
itself, a condition demoted for not paying rent, a conjecture that became a
fact only after settling true, and a Goodhart collapse found by name.

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
- **Two context prefixes are reserved.** Keys starting with `do:` mean the
  agent was *acting on* the world (interventions) — a guard found on one is
  labeled regime dependence and mixed-regime predictions carry a
  `regime_mixed` caveat. Keys starting with `derived:` belong to the sweep's
  synthesized frames and are never yours to write.
- **Admitted conditions can be demoted.** A guard whose direction reverses on
  post-admission data — or stops beating chance on twice the evidence that
  admitted it — is removed through the ledger, and it only re-enters on fresh
  post-demotion evidence. Watch for `status: "demoted"` entries in
  `run_gate()`'s output.
- **Deleting `index.sqlite3` loses nothing.** It is a derived view; `replay()`
  rebuilds it bit-for-bit from the ledger segments.
- **To undo a bad source, use `retract_source`, not `redact`.** Payloads are
  content-addressed and carry no actor, so two sources reporting the same
  outcome on the same statement share one — and `redact` purges *content*,
  meaning it takes the honest reports with it. `retract_source` is scoped to
  the actor, is reversible, and recomputes every downstream number as if that
  source had never spoken. `redact` is for secrets and PII; check
  `redaction_scope()` before firing it.
- **Quotas are on by default.** An actor that floods observations or spams the
  gate hits its per-epoch quota and gets a `QuotaExceeded` — provision with
  `set_actor_quota()` for legitimate bulk loads. The quota rolls over each epoch,
  so a well-behaved actor is never permanently locked out.
- **`authority` is a label, not a login.** By default the `actor`/`authority` on
  a write is attribution, not an authenticated identity — CANDOR's trust
  boundary is the process. If you need access control, register a policy with
  `set_authz(...)` and the privileged writes (pin, redact, retract_source,
  register_oracle, set_reliability) are enforced, raising `Unauthorized` before
  anything is appended. See [SECURITY.md](../SECURITY.md).

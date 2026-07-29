# CANDOR

**A memory substrate for AI agents that knows why it believes things.**

Most agent memory is a vector store: text goes in, similar text comes out, and
"confidence" is a cosine score that means nothing. CANDOR is built on a
different bet — that an agent's memory should work like a careful scientist's
lab notebook:

- **Every belief traces to evidence.** Facts are backed by attributed
  observations in an append-only ledger. Ask `why(fact)` and you get the whole
  chain: who claimed it, what the gate checked, every count that moved.
- **Probabilities are earned, not vibes.** When CANDOR says *0.83*, that
  number comes from observed outcomes, per-source trust learned from settled
  predictions, and calibration you can audit — never from "the embedding was
  close."
- **Sources earn trust by being right.** Every observation is keyed by who
  made it. When predictions settle against reality, each source's confusion
  ledger updates — so an always-agreeable source's "yes" ends up worth
  nothing, while a careful checker's rare "no" becomes decisive.
- **Instability is a clue, not noise.** A fact that flips isn't "uncertain" —
  it's a function with a missing argument. CANDOR detects overdispersion,
  hunts the missing variable, and proposes *conditions* ("works **when**
  method=crawl4ai"). When something changes for good, it finds the date:
  regime changes are located, not decayed away.
- **It knows what it hasn't seen.** Beyond true/false, a fact can record an
  *open-vocabulary* categorical outcome (which of many values happened), and the
  vocabulary grows as new values appear. The distribution carries a first-class
  *unknown* mass — "captcha 73%, block 18%, and 9% a value we've never seen" —
  so an unfamiliar outcome is a real possibility with its own probability, not a
  rounding error.
- **Nothing is ever silently mutated.** Change is append + recompute. Delete
  the SQLite index entirely and rebuild it bit-for-bit from the log. Retract a
  poisoned source and every downstream number recomputes as if it never spoke.

It found real things on its first contact with lived data: replaying an
agent's operational history, it located the exact date a broken tool was fixed
(0% → 79% success) and the week web-search reliability collapsed (93% → 38%) —
both independently corroborated by the agent's own notes, neither planted.
It also *refused* two plausible-looking rules that failed held-out validation,
which matters just as much.

## Quickstart

```sh
git clone <this repo> && cd candor
make venv            # python3.11+; installs pytest + hypothesis (test-only deps)
make gates           # run the conformance stage gates
.venv/bin/python examples/quickstart.py
```

The core has **zero runtime dependencies** — standard library only.

```python
from candor.system import CandorSystem

m = CandorSystem("./store")

# Everything enters as a candidate and passes a 7-step admission gate.
m.assert_({"pred": "flaky_api", "args": ["payments"], "stmt_type": "frequency"},
          source="runbook", actor="human:me")
m.run_gate()

# Observations are attributed and context-rich. Log wide: ambient context is
# how missing variables get found later.
m.observe({"pred": "flaky_api", "args": ["payments"]}, outcome=False,
          ctx={"region": "us-east", "hour": "03"}, actor="tool:probe")

p = m.predict({"pred": "flaky_api", "args": ["payments"]}, budget=10_000)
print(p.p, p.ci, p.snapshot_id)   # a real probability, reproducible from its snapshot
```

## What's in the box

| Path | What it is |
|---|---|
| `SPEC.md` | The full v0.2 design spec — invariants, data model, test harness |
| `docs/` | Architecture, API guide, use cases, benchmark story, spec deltas |
| `examples/` | Runnable, offline demos of the main workflows |
| `src/candor/core/` | Trusted core: ledger, gate, closure, counts, calibration (stdlib-only) |
| `src/candor/periphery/` | Untrusted periphery: retrieval, prediction, extraction, curiosity |
| `tests/conformance.py` | The executable spec — 23 conformance tests across 5 stage gates |
| `tests/unit/` | 443 additive tests |
| `tests/claims/` | The executable README — every claim on this page, measured on synthetic worlds with planted truth and null controls (`make claims`) |
| `bench/` | The pre-registered honest-test harness and its findings |
| `DEVIATIONS.md` | Every place the build interprets, extends, or argues with the spec |

## The honest part

This project graded itself in public, with pre-registered margins frozen
before each run. Three rounds against a strong baseline (dense retrieval +
elicited probabilities from a 27B model): **retrieval passes and wins
outright; the trust machinery decisively beats its own machinery-off control;
calibration honesty (slope, ECE, log loss) leads every round — and the
Brier-vs-fresh-reader bar failed all three times**, which stands on the record
as the documented trade (stored sparse witnesses can't outscore the strongest
judge reading everything fresh at query time). The whole story, including the
failures and what each one taught, is in [docs/benchmarks.md](docs/benchmarks.md).

## Learn more

- **[Getting started](docs/getting-started.md)** — install, first store, the core loop
- **[Architecture](docs/architecture.md)** — the ledger, the gate, trusted vs untrusted, the invariants
- **[API guide](docs/api.md)** — every call with examples
- **[Use cases](docs/use-cases.md)** — agent memory, source-reliability tracking, drift detection
- **[Benchmarks](docs/benchmarks.md)** — the pre-registered 6.8 rounds, honestly told
- **[Security model](SECURITY.md)** — the trust boundary, opt-in access control, and what the hash chain does and doesn't prove

## Status

Fully conformant against the spec through Stage 5 (23 conformance gates green),
backed by 443 additive unit tests (1 xfail-by-design) and 35 claims tests
holding this page to its word. Spec v0.2 plus adopted deltas v0.3 / v0.4 / v0.5
(v0.5 adds open-vocabulary categorical facts with a first-class unknown mass and
read-time distribution surfacing). Single-writer, single-box by design —
distributed consensus is an explicit non-goal for v1.

Six defects have been found by the claims suite and fixed, each recorded with
before/after measurements in [bench/CLAIMS_HARDENING.md](bench/CLAIMS_HARDENING.md)
— including a purge path that destroyed honest observations along with the bad
source, and an overdispersion statistic that fabricated conditions for
mostly-failing tools 41% of the time. Conformance passing is not the same as
the README being true, which is why the second suite exists.

## License

[MIT](LICENSE).

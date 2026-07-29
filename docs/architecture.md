# Architecture

## The one-sentence version

An append-only, hash-chained event ledger is the only primary artifact;
everything else — SQLite tables, closures, probabilities, trust scores — is a
derived view that can be deleted and recomputed, which is what makes every
number auditable and every mistake reversible.

## Physical layout

```
store/
  ledger/segments/00000N.jsonl   hash-chained event skeletons (append-only)
  ledger/payloads/<sha>.json     content-addressed payloads (deletable = redaction)
  index.sqlite3                  derived view — drop it, replay(), identical
  evidence/*.md                  prose tier: entries with @salience, stable ids
  retrieval.sqlite3              retrieval side-stream log (outside the chain)
```

The chain covers payload *commitments*, not payloads — so redacting a payload
(deleting the file) leaves the chain verifying while replay recomputes all
downstream state *without* the redacted content. Torn writes recover by
truncating to the last verifying line: loss, never silent corruption.

The chain is a **consistency** mechanism, not a **tamper-evidence** one. Each
event commits to its predecessor's hash, so `verify_chain()` catches accidental
corruption, torn writes, and two processes forking the same log — the failure
modes that lose or scramble data. It does *not* prove the log was never
rewritten: anyone who can edit a segment can recompute every hash after their
edit and the chain will verify. Tamper-evidence against a writer with that
access needs an external anchor (a signed or notarised head), which CANDOR
leaves to the deployment — see [SECURITY.md](../SECURITY.md).

## Trusted core vs untrusted periphery

LCF-style separation, enforced physically by the source tree:

- **`src/candor/core/`** (standard library only, imports nothing from
  periphery): ledger, canonicalizer, gate harness, closure engine, proof
  kernel, count updater, reliability scorer, calibration. Small enough to be
  reviewed; the proof kernel is small enough to be *believed*.
- **`src/candor/periphery/`** (fallible, always gated, always attributed):
  retrieval, the prediction engine, the extractor, the curiosity sweep, the
  optional dense embedder. Anything here can be wrong without corrupting the
  store, because nothing here can write a committed number.

Two firewalls are checked mechanically, not by convention (`make audit`):
the retrieval module has an empty import list (so no code path exists from
retrieval to the count updater), and the token "weight" does not appear
outside the committed tier.

## The invariants that hold it up

| | Invariant | Why it matters |
|---|---|---|
| I1 | Ledger is the only primary artifact | rebuild-from-log = audit + recovery for free |
| I2 | Retrieval never moves a number | otherwise memory manufactures its own consensus |
| I3 | Nothing is mutated; change = append + recompute | "crystalline" is a permission set, not a place |
| I5 | Zero is reserved for refutation; unobserved gets ε | hard zeros are absorbing states |
| I7 | Epistemic and aleatoric uncertainty never share a number | they compose differently through inference |
| I8 | Predictions record a model snapshot; re-running reproduces them exactly | silent recalibration contaminates history |
| I11 | Storage holds integer counts keyed by (fact, actor, channel); every real number is read-time | a discounted count can be recomputed forever; a mutated one cannot |

## Two channels, four axes

Every fact carries a `stmt_type` that decides what an observation means:

- **crisp** — "X is true": observations move the *epistemic* channel
  (belief about truth).
- **frequency** — "X happens at some rate": observations are trials and move
  the *aleatoric* channel (the rate); belief in the reference class itself
  moves only through structural events.
- **categorical** (v0.5) — "X takes one of an open set of values": each
  observation records a *value*, the vocabulary grows as new values appear, and
  the read-time predictive is a distribution over the seen values plus a
  first-class *unknown* mass (Dirichlet-process / CRP, `alpha/(N+alpha)`). It is
  neither channel — it is a distribution — and it composes with the same
  curiosity and settlement machinery via a per-value one-vs-rest reduction. See
  [spec-v0.5-delta.md](spec-v0.5-delta.md).

Orthogonally, a fact has `structural` status (candidate/admitted/pinned),
`numeric` status (accumulating/frozen), and an admission `kind`
(exact/soft/definitional). A rule can be structurally permanent while its
reliability floats; a fact can be currently certain and structurally
provisional.

## Trust: the two-coin model (v0.3–v0.4)

Per (actor, frame), the store keeps an integer **confusion ledger** — and with
graded observations, a categorical **response ledger** — moved *only* when a
claim settles through a deterministic oracle. At read time a vote's evidence
is its likelihood ratio. This is what lets asymmetric sources be exactly what
they are: an always-yes source's "yes" carries LR ≈ 1 (no information) while
its rare "no" is decisive; a lexical checker's "absent" can be near-decisive
while its "present" is weak. Votes sharing a context signature compose
sub-additively (correlated evidence is priced, not double-counted). A single
"reliability score" cannot represent any of this — we proved that the hard
way; see [benchmarks.md](benchmarks.md).

## Prediction

Model counting over the proof DNF inside an epistemic outer loop: facts and
actor parameters are sampled once per epistemic world, so uncertainty
correlates across queries that share a fact or an observer — by construction.
Sampling is stratified and deterministic (inverse-CDF quantiles dealt through
identity-derived permutations), which makes three conformance properties true
of the *estimator* rather than of Monte Carlo luck: monotonicity in support,
insertion-order invariance, and bit-exact snapshot reproduction. Constraint-
violating worlds get zero mass and the rejection rate is reported as a health
signal. Recalibration is a read-time isotonic map whose hash rides in every
snapshot.

## The curiosity engine (Stage 5)

A sweep over the observation log that treats instability as information:

- **Overdispersion** (Tarone's Z, BH-corrected across the context keys tested)
  → a *guard* candidate ("true when key=value"), which must clear per-side
  support, an MDL check, and held-out validation at the gate. High-cardinality
  keys can flag a problem but can never become guards — an m-ary lookup table
  is not a condition.
- **Derived frames** (v0.6 Δ10) — the keys tested are not limited to what the
  agent logged. The sweep synthesizes candidate frames from data already in
  the ledger — `derived:hour`/`derived:dow` from event timestamps,
  `derived:prev` (the fact's own previous outcome), and pairwise interactions
  of recorded keys — and runs the identical machinery over them. Recorded keys
  outrank derived ones at winner selection, and `derived:prev` may only win
  when it *absorbs* the time structure it claims to explain (a one-way step
  routes to the located date instead; a residual block pattern routes to the
  open question). Breadth stays recorded-only: synthesized diversity is not
  logging diversity.
- **One-way change** — located at the argmax of cumulative deviation (median
  error 1 observation in 120), then tested with a two-sided Fisher exact
  p-value corrected for the split positions searched, plus the same test inside
  each segment so oscillation isn't mistaken for a step. Exactness earns its
  keep here: a CUSUM normalised by `sqrt(p(1-p))` is a Gaussian approximation,
  and skewed Bernoulli increments made the previous version fire on 40% of
  *stationary* p=0.95 segments while eating most real breaks. Survivors become a
  *supersede-with-valid-time* candidate carrying the located date; the old
  regime keeps its counts and its dates, the successor starts fresh, and the
  gate checks the date, the per-side support and the significance like any other
  candidate. This is the substrate's answer to recency: **age isn't decay, it's
  regimes.**
- **Detected but unexplained** → an open question with the residual partition
  and a concrete suggested measurement. Instability is tested on the time axis
  as well as across covariates, so a stream that swings with *nothing useful
  logged* is told "log wider" instead of getting silence. When a new covariate
  starts being recorded later, open questions are re-tested against it
  automatically.
- **The prospective audit** (v0.6 Δ11) — admission is not tenure. `run_gate()`
  opens by scoring every admitted guard on the observations that arrived
  *after* its admission, and demotes it through the ledger when its direction
  reverses (§3.4 hysteresis, compared in evidence-nats) or goes stale on twice
  the entry evidence. The demoted rule leaves the closure and every read path;
  the memoryless sweep's re-proposal is then judged on *post-demotion*
  evidence only, so a dead guard cannot flap back in on the history that
  admitted it — but a world that re-structures can earn it back.
- **Intervention labeling** (v0.6 Δ13) — context keys prefixed `do:` mark the
  agent *acting* rather than watching. A guard on one is labeled regime
  dependence (P(·|observe) ≠ P(·|do)) rather than a mere condition, and
  predictions pooling across mixed `do:` regimes carry a `regime_mixed`
  caveat. Vocabulary, deliberately not a causal model: nothing predicts what
  an intervention will change before post-intervention data exists.

**Timing and `closure_hash`.** The sweep is batch-triggered, not
per-observation: `run_gate()` runs it, and a reopen re-runs it while folding.
Between a bare `observe()` and the next `run_gate()` or reopen, the derived
*sweep* outputs (a fact's breadth and dispersion flags) still reflect the
previous sweep, so `closure_hash()` in that window is a faithful snapshot of a
deliberately stale sweep — true of crisp, frequency, and categorical facts
alike. Live state equals replayed state *after* any `run_gate()` or reopen,
which is exactly where the I3 equivalence is asserted; call `run_gate()` before
comparing closure hashes across processes.

## The conjecture loop (v0.6 Δ12)

Soft unification proposes analogies over *behavioural* signatures (which rules
a symbol fires in, what it co-derives with) — never text similarity, and never
typed as a proof. v0.6 closes the loop that used to dead-end at the proposal:
`conjecture(goal, sim_budget, commit=True)` files each analogy as a **claim**
by `agent:conjecture`, whose predicted probability is the analog's own earned
number transferred across the edge (similarity is a licence, not a
probability). The claim calibrates under its own `conjecture/v1` predictor
class — the analogy engine's track record is a measured curve, never pooled
with the prediction engine's — and only a claim that *settles true* asserts
the goal as a fact candidate, through the same gate as everything else.
Postulate, validate, implement — in that order, never out of it.

## Spec lineage

`SPEC.md` is the frozen v0.2 spec. [spec-v0.3-delta.md](spec-v0.3-delta.md)
(two-coin trust, context-grouped composition, permanent control, dense
retrieval as periphery input), [spec-v0.4-delta.md](spec-v0.4-delta.md)
(graded observations, witness floor), [spec-v0.5-delta.md](spec-v0.5-delta.md)
(open-vocabulary categorical facts with a first-class unknown mass, read-time
distribution surfacing), and [spec-v0.6-delta.md](spec-v0.6-delta.md) (derived
context frames, the prospective guard audit, committed conjectures, `do:`
intervention semantics — with the E→A boundary stated and tested) were adopted
after pre-registered test rounds; each delta cites the evidence that forced it.
`DEVIATIONS.md` records every interpretive decision.

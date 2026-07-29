# CANDOR — spec v0.6 delta

Amends v0.5. Motivated by an outside document — an axiomatization of the
"LLMs can't jump" argument (Peirce's triad; scientific invention as abduction;
the E→A step that produces axioms rather than consuming them) — read against
this codebase. The reading found that CANDOR already runs one full
postulate→validate→implement loop (curiosity → held-out → gate), but with three
honest gaps: it could only postulate over variables somebody logged (the
"parameter pond" / frame-selection wall), its validation was retrospective-only
(a guard admitted once was admitted forever), and two loops were half-built
dead ends (conjectures returned to the caller and vanished; intervention was
indistinguishable from observation). The four deltas below close what is
closable *observationally* and label what is not.

The boundary is stated up front, because the source document is right about it:
none of this mechanizes the E→A jump. Δ10 widens the hypothesis space by
composition over an internal basis — it extends the pond, it does not leap
between ponds. Δ13 gives intervention a vocabulary, not a causal model: the
system can now *say* "this coupling was regime-dependent" after the fact, and
it still cannot predict what an intervention will change before
post-intervention data exists. That wall is asserted as expected behavior in
`tests/claims/test_axiom_battery.py`, so a future system that passes it is
measurably different from this one.

## Δ10 — derived context frames (the sweep postulates variables nobody logged)

The covariate sweep conditioned only on recorded `obs_context` keys. It now
augments each fact's observations with SYNTHESIZED frames, all pure functions
of data already in the ledger:

  * `derived:hour` / `derived:dow` — UTC clock frames from the event ts.
    Periodic structure is invisible to the changepoint/temporal machinery,
    which sees level and spread but never phase.
  * `derived:prev` — the fact's own previous outcome ("T"/"F", or the previous
    categorical value). State dependence: sticky caches, flapping services.
  * `derived:{k1}x{k2}` — pairwise interactions of recorded keys, synthesized
    only when the fact records ≤ 4 distinct keys. The bound is measured, not
    guessed: at 6, the preregistered weak-covariate world (0.70-vs-0.40 with
    five nuisance keys) gains 15 null pair tests and its recovery rate falls
    below the claims-suite floor — O(K²) pairs dilute the BH budget faster
    than they add power.

Everything downstream is UNCHANGED: Tarone → BH → MDL → held-out quarter →
gate. Three rules keep the new frames honest:

  * **Recorded outranks derived** at winner selection — the agent's own
    vocabulary is the primary explanation space; a synthesized frame speaks
    only when nothing logged explains the variance.
  * **`derived:prev` must absorb what it claims to explain.** Self-lag is the
    frame most prone to shadowing other structure: a one-way step makes
    prev≈current by construction (the honest repair is the located DATE — §4.4
    routing is computed first and prev yields to it), and an unlogged block
    variable makes prev a smeared proxy (so prev is disqualified unless both
    prev-conditioned subseries are time-stable — no residual temporal
    dispersion, no leftover step). A genuine sticky process passes both.
  * **Breadth ignores derived frames** — breadth measures the agent's logging
    diversity, and a synthesized hour must not inflate it.

Purity: the augmentation is a per-fact pure function of the log, so the resweep
contract (H6b), checkpoints, `predict_at` and replay hold verbatim. Read-side,
`distribution()` gains an additive `derived_modes` section and computes η² on
the augmented projection when the admitted guard's key is derived; `modes`
stays recorded-only, keeping its docstring's promise.

Evidence: `tests/unit/test_derived_keys.py`.

## Δ11 — the prospective audit (admitted guards keep paying rent)

Entry validation was retrospective only: the held-out quarter existed at
proposal time, and an admitted guard was admitted forever — the sweep's
re-proposals of a failed guard were new candidates judged against the same old
history, while the original stood. Now `run_gate()` opens by scoring every
admitted guard's direction on the observations that arrived AFTER its
admission — predictions it actually risked — and demotes through the ledger
when either signal fires:

  * **Reversal**: the post-admission record supports the inverted direction at
    the §3.4 hysteresis bar. Evidence is compared in NATS
    (`gate.direction_evidence`: signed binomial log-likelihood vs chance),
    which grows with sample size — a count-odds ratio saturates and would make
    any hysteresis bar unreachable.
  * **Staleness**: the direction stops beating chance (hits ≤ misses) on twice
    the audit floor — removal on a null effect needs strictly more evidence
    than entry did.

The demotion event (kind `demotion`, already in the fold's vocabulary — this
delta is what finally appends one) marks the rule `candidate` (out of the
closure and every read path) and closes the candidate row as `demoted`,
recording the against-evidence and the demotion's own event seq.

**Anti-flap.** The sweep is memoryless, so it re-proposes the demoted guard
from the very history that admitted it, and a full-history holdout is dominated
by the era when the guard genuinely worked. Re-entry is therefore judged on
FRESH evidence only: the harness attaches the guard's post-demotion record
(`body["post_demotion"]`, scored by the same machinery as the audit) before
gate evaluation, and the gate requires it to overcome the demotion's
against-evidence by `DEMOTION_HYSTERESIS`. No scorable post-demotion record
yet → no re-entry yet. A demotion is not a life sentence: when the world
re-structures, the fresh record clears the bar and the guard returns.

Placement: the audit CANNOT live in the sweep — it reads gate state, and the
sweep's verdicts must stay pure functions of each fact's own observations
(H6b). It lives in `run_gate` (the §3.4 serialization point) behind its own
observation watermark; the gate decides, the decision is appended, and replay
folds it without re-judging. Untrusted periphery computes the score; the
trusted side owns the decision.

Schema: `candidates.status` gains `'demoted'` (the index is derived — I3 — so
this is a rebuild, not a migration).

Evidence: `tests/unit/test_guard_prospective_audit.py`.

## Δ12 — conjecture claims (the analogy loop closed)

`conjecture()` was a dead end: proposals returned to the caller and nothing
consumed them, despite the module docstring's promise ("observation tests; the
gate promotes survivors"). `conjecture(goal, sim_budget, commit=True)` now
files each proposal as a CLAIM by `agent:conjecture`:

  * `predicted_p` is the ANALOG's own earned probability — `predict(via).p` —
    transferred across the behavioral-similarity edge. Similarity is not a
    probability; the transfer moves an audited number, and `sim` rides the
    payload for audit. (A categorical analog is proposed but not committed: a
    distribution is not a transferable scalar.)
  * The claim carries the distinct predictor class `conjecture/v1`, so the
    analogy engine calibrates on its own curve, never pooled with the WMC or
    categorical paths (I9). That curve IS "how much is an analogy worth
    here" — measured, not presumed.
  * `resolve(claim, outcome=True)` auto-asserts the goal as an ordinary fact
    candidate (source `conjecture:<claim_id>`, proposer `agent:conjecture`)
    THROUGH the gate — I10, through, never around. A false settlement
    implements nothing; either way the curve was scored.
  * An unresolved conjecture claim for the same statement is not re-filed
    (idempotent commit); a resolved one may be re-conjectured.

I4 intact throughout: a conjecture is never typed as a Proof, the read-only
path is byte-identical, and no committed number moves until observations do.

Evidence: `tests/unit/test_conjecture_claims.py`.

## Δ13 — `do:` intervention semantics (acting is not watching)

A reserved context-key prefix, `do:`, records that the agent (or its
principal) was acting on the world rather than observing it. No new
statistics — a `do:` key partitions like any covariate — but the semantics
change at exactly two points:

  * A guard conditioned on a `do:` key carries `regime_dependent: true` and
    its open question says so: the coupling holds within one intervention
    regime and DOES NOT TRANSFER across `do(...)` — P(·|observe) ≠ P(·|do),
    which is the Goodhart structure by name.
  * A prediction pooling observations across mixed regimes (two `do:` values,
    or partial coverage — absent IS a regime) carries a `regime_mixed` caveat
    on both the scalar and categorical paths. The marginal is still reported —
    predictions are unconditional by architecture — but it announces that it
    averages over a boundary; the per-regime numbers live in `distribution()`.

Stated limit: this is vocabulary, not causality. Detection is post-hoc by
construction; predicting a decoupling BEFORE post-intervention data exists
requires intervention access (a causal model or a world to act in), not a
better statistic over these observations. The battery asserts that wall.

Evidence: `tests/unit/test_do_semantics.py`;
`tests/claims/test_axiom_battery.py` (EX-7 marked/unmarked/boundary).

## The battery

`tests/claims/exemplars.py` + `tests/claims/test_axiom_battery.py` render the
source document's §9 exemplars (Parkinson, Peter, Goodhart, Dunning–Kruger) as
planted worlds and assert the C-16 differential profile: LC-2 structure
originated as gated conditions; LC-3 named when the intervention is marked,
dated when it is not, and never predicted in advance; the C-18
regression-to-the-mean artifact DECLINED — the negative control counting as
much as any hit.

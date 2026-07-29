# DEVIATIONS

Every place where this build departs from, disambiguates, or adds to
`SPEC.md` (CANDOR v0.2), plus every edit to the conformance harness.

The spec is authoritative; the harness (`tests/conformance.py`) is executable
spec. Where they conflict, the spec wins and the conflict is recorded here.

---

## Edits to the conformance harness

The harness is otherwise byte-identical to the delivered `candor_conformance.py`.
No assertion has been weakened, skipped, or removed.

### D1 — `make_driver()` wired to the implementation
`tests/conformance.py`, the body of `make_driver()`. The shipped body was
`pytest.skip("wire make_driver() to your HarnessDriver implementation")`, which
the file's own docstring designates as the wiring point. It now builds
`candor.harness.CandorHarnessDriver`, passing the harness's own result types
(`DeriveStatus`, `DeriveResult`, `Prediction`, `RawCounts`, `ComposedCounts`)
into the adapter. They are injected rather than imported by the implementation
so the harness stays the single definition of the contract — the tests compare
`DeriveStatus` members by identity, so the driver must return the harness's
enum, not a look-alike.

**This is the only edit to the file.** `diff candor_conformance.py tests/conformance.py`
shows exactly this hunk.

### D2 — stage markers registered from `conftest.py`, not from the harness
The harness defines `pytest_configure` to register the `stage1..stage5`,
`fail_stop` and `alert_only` markers. pytest only calls that hook from
`conftest.py` files and plugins, never from a test module, so as shipped the
markers were never registered. Rather than move the function (which would edit
the harness), the same registration lives in `tests/conftest.py` and in
`pytest.ini`. `tests/conftest.py` also puts `src/` on `sys.path`. No assertion
is involved. The harness's own `pytest_configure` is left in place, verbatim
and inert.

---

## Spec / harness discrepancies

### D3 — gate step 7 vs. constraint conditioning (§3.4 step 7 vs. §3.9)
**The conflict.** §3.4 step 7 says admission requires that `closure ∪ candidate`
"violate no admitted constraint". Read maximally, a second mutex-exclusive fact
can never be admitted. But §3.9 requires the prediction engine to reject
constraint-violating epistemic worlds and to *report a nonzero
`rejection_rate`*, and §6.2 "constraint conditioning" tests exactly that by
asserting two mutex-exclusive facts through the gate and demanding
`rejection_rate > 0`. Under the maximal reading the second fact never exists,
`rejection_rate` is identically zero, and §3.9's rejection accounting is
unreachable dead code.

**Resolution (spec wins, ambiguity resolved toward the reading that keeps both
mechanisms live).** Step 7 is evaluated against the **certain fragment** of the
closure: `definitional` facts, `pinned` structures, and facts carrying a `+`
pin. A candidate contradicting any of those is rejected at step 7 and the
rejection is recorded with the failing step. Tension between two merely
*admitted* facts — which are epistemically uncertain by construction, since
admission does not assert certainty — is admitted, and priced at read time as
`rejection_rate` in `predict`, exactly as §3.9 specifies.

Implemented in `core/gate.py::_certain_contradiction` and
`core/apply.py::rebuild_closure`. Both readings are tested:
`tests/unit/test_gate_steps.py::test_step7_contradiction_against_a_certain_fact`
(rejection) and `tests/unit/test_predict.py::test_mutex_group_produces_rejections_and_renormalizes`
(conditioning).

**Not load-bearing enough to block on**, because the harness settles it: the
only executable statement of the requirement demands a nonzero rejection rate.

### D4 — scope of the lexical firewall (§3.2, §6.2)
The spec says "`grep -rn weight` outside the committed tier returns nothing".
Taken literally against this repository that can never pass, because the audit
itself must name what it looks for, and the hook name
`grep_weight_outside_committed` is fixed by the immutable `HarnessDriver`
protocol.

The scan (`candor/audit.py`) therefore covers **all** of `src/candor/**/*.py`
plus the evidence tier's `.md` files, and exempts exactly:

* `src/candor/core/committed/` — the committed tier, where weights belong;
* `src/candor/audit.py` and `src/candor/harness.py` — the auditor and the
  test-only driver adapter. Neither is part of the substrate; `harness.py` is a
  pure delegation and never ships in a production surface.

The exemption list is itself asserted in
`tests/unit/test_audit.py::test_exemptions_are_exactly_the_audit_surface`, and
the scan is proved non-vacuous by a negative control that plants `@weight:` in
an evidence entry and requires it to be reported. Extending the scan to the
evidence tier is an *addition* to the spec's letter, in the direction of its
intent: `@weight → @salience` is the rename the firewall exists to protect.

The prediction engine's prose was reworded from "weighted model counting" to
"model counting over literal masses"; the code already spoke in terms of
`mass`, so nothing was renamed to dodge the check.

### D5 — §6.7 stage gates name tags the harness does not implement
§6.7 requires these tags green that have no corresponding test in
`candor_conformance.py`:

| Stage | Tag named in §6.7 | Present in harness? |
|---|---|---|
| 1 | calibration loop (trivial weights) | no |
| 2 | premise entailment · gate steps · canonicalization | canonicalization only |
| 3 | (all present) | — |
| 4 | monotonicity · irrelevance · permutation | irrelevance missing |
| 5 | MDL/BH fixtures · breadth discrimination · valid-time isolation | no |

The harness is the definition of done for the gate, so a stage exits on its
tagged harness subset. The missing behaviours are nevertheless implemented and
covered additively under `tests/unit/` — premise entailment in
`test_kernel_and_closure.py`, all seven gate steps in `test_gate_steps.py`,
the calibration loop in `test_calibration.py`, MDL/BH/breadth in
`test_curiosity_stats.py`, and `irrelevance` — named in §6.2 and in the Stage-4
gate but absent from the harness — in `test_properties.py`.

### D6 — `test_a_reputation_farming` is an unimplemented xfail
The harness body is `raise NotImplementedError(...)` under
`@pytest.mark.xfail(reason="reputation farming: v1 documented limitation")`. It
is left exactly as delivered. Per §6.5 it is `xfail-by-design`, and per the
run's scope xfail counts as pass for this probe only. Writing the real probe —
which must pin the blast radius at quota × discount — is Stage-3 follow-up work
that the harness does not currently require.

---

## Implementation decisions the spec leaves open

These are not deviations; they are choices the spec delegates. Recorded because
they are load-bearing for the numbers.

### D7 — deterministic stratified sampling instead of pseudo-random draws
§3.9 specifies `for s in 1..S` epistemic samples with `v_f ~ Beta(epi_f)` etc.
This build draws the S values as the **inverse-CDF quantiles at
`(i+0.5)/S`**, dealt to sample indices through a permutation derived from the
fact's own id (splitmix64 Fisher–Yates). The marginal law per fact is the
intended Beta; the draw is reproducible and, crucially, *monotone in the
success count*.

Why it matters: three §6.2 invariants are otherwise statements about Monte
Carlo luck rather than about the estimator —

* **snapshot completeness (I8, fail-stop)** demands `predicted_p` reproduce
  exactly, to `abs=1e-12`;
* **permutation** demands two insertion orders of the same observations agree
  to `abs=1e-9`;
* **monotonicity** demands added support never lower `p`.

With this scheme all three hold by construction. `predict` is a pure function of
the composed posteriors, so it is independent of the ledger head hash and of
insertion order.

### D8 — the epistemic prior of an admitted fact is Beta(99, 1)
§4.2 says a `frequency` fact's epi channel "moves only through *structural*
events — never through trials", and §6.2 two-channel routing forbids any epi
count on a frequency fact. So a freshly admitted frequency fact's validity
posterior comes entirely from its prior. It is set to Beta(99, 1)
(`E[v] = 0.99`) in `core/committed/counts.py`, meaning: passing all seven gate
steps is itself strong evidence the reference class is valid as stated, but not
proof. Consequences: `predict` on an unobserved frequency fact reports ≈ 0.495,
and the reported CI widens slightly at the bottom because ~1% of epistemic
worlds carry `t_f = 0`. Measured coverage on the §6.4 synthetic world is 28/30
(threshold 26.1/30) with a mean CI width of 0.224 — an interval that is
conservative but not degenerate.

Aleatoric ε-priors are uniform Dirichlet, `Beta(1, 1)` (I5: unobserved gets ε,
never a hard zero).

### D9 — default actor reliability is Beta(19, 1)
§3.12 defines the discount but not the prior for an unscored actor. It is
`E[rel] = 0.95`: the discount is a penalty for demonstrated unreliability, not
a tax on newcomers. A default of 0.5 would shrink every posterior toward the
midpoint before any actor had done anything wrong.

### D10 — zero count rows are created at admission
`fact_counts` gets a `(fact, proposer, channel) = (0, 0)` row when a fact is
admitted, so an unobserved admitted fact has an addressable audit trail (I5's
"unobserved gets ε" made concrete in storage as a zero-count row, with the
ε itself supplied at read time by the priors). This is what makes
`raw_counts(fid)` non-empty before any observation, which the harness's alias
reversibility test requires.

### D11 — logical structure of stored ids
Fact ids are content addresses over the *canonicalized* `(pred, args)`
(`fact:<sha256[:32]>`), so identity survives replay, process restarts and index
drops without a sequence counter. Candidate, gate-run, rule, constraint and pin
ids are derived from the ledger sequence number of the event that created them,
which is likewise replay-stable.

### D12 — reliability overrides are persisted outside the chain
`set_reliability` is a test-only `HarnessDriver` hook that sets a posterior
directly, bypassing §3.12's trusted-settlement path. Since the reliability
table is part of the closure hash, an override that lived only in SQLite would
break replay determinism. Overrides are written to
`<root>/reliability_overrides.json` and re-applied after every rebuild. The
production path (`resolution` events scoring against a `deterministic_total`
oracle) writes through the ledger and needs no such file.

### D13 — `recall` searches the payload store as well as the evidence tier
§5 defines `recall` over the evidence tier. This build also searches the
content-addressed payload store, presented as source-material spans. It is what
makes §6.6 redaction replay non-vacuous: before redaction `recall("purge-me")`
returns the observation payload, after it the file is gone and the query returns
nothing. Both corpora are read-only and both go through the same side-stream
log; `periphery/retrieval.py` imports nothing from the package (I2).

### D14 — gate decisions are ledger events, and rejections are events too
§3.4 says the gate "emits an `admission` event with a `gate_run_id`" and that
"rejections are recorded, not discarded". `admission` is not a permitted event
kind for a rejection under a strict reading of the §2 CHECK set, and no
`rejection` kind exists. This build emits one `admission` event per candidate
decision, carrying `status ∈ {admitted, rejected}` and the failing step in the
payload. Replay therefore reproduces the committed tier without re-running any
judgement — the gate decides, the log records, the replayer applies.

### D16 — the epistemic prior is not shared between statement types
**Supersedes the second half of D8.** D8 gave every admitted fact an epistemic
prior of Beta(99, 1). Running the system against a real corpus showed that to be
wrong for `crisp` facts, and wrong in a way §4.2 already rules out:

| stmt_type | what epi means (§4.2) | is admission evidence for it? |
|---|---|---|
| `frequency` | p(the reference class is valid as stated) | **yes** — and it is the only support, since trials move alea and never epi |
| `crisp` | p(the statement is true) | **no** — every gate step is a structural or consistency check; none of them observes the world |

Under the shared prior the gate was quietly asserting that admitted crisp facts
were 99% true, and it took ~100 contradicting observations to drag such a fact
back to even odds. Now `EPI_PRIOR_CRISP = (1, 1)` and
`EPI_PRIOR_FREQUENCY = (99, 1)`. The §6.4 coverage result is unchanged (the
synthetic world is all frequency facts); `test_p_constraint_conditioning` still
passes, with the mutex rejection rate moving from ~0.98 to ~0.25 because two
uncertain crisp facts are now genuinely uncertain. Covered by
`tests/unit/test_counts_and_channels.py::test_the_epistemic_prior_is_not_shared_between_statement_types`.

### D17 — a registered oracle outranks the closure in settlement triage
§3.8 triages a claim as `entailed` when it "follows from committed facts by
exact derivation". Taken first, that rule makes external settlement unreachable
for anything already admitted — and since admission is structural, *every*
admitted statement qualifies. §3.8's own warning is that self-consistency is not
truth and that the external-settled ratio needs a floor, so `_triage` now checks
for an explicitly named, registered oracle *before* trying the closure. A caller
that has constructed an external verifier gets external settlement; everything
else is unchanged.

### D18 — the §6.8 bench harness, and where embeddings are allowed to live
§0 forbids text embeddings "anywhere in the core", but §6.8 requires comparing
against an embedding store and against RAG with elicited probabilities. Both
comparators therefore live in `bench/`, outside `src/candor/`, and the
dependency runs one way only: `bench → candor`, never the reverse. Asserted by
`tests/unit/test_audit.py`. `bge-m3` and `laguna-s-2.1` are reached over HTTP
from the bench side; the substrate has no runtime dependencies and still
contains no embedding of any kind.

The real corpus also exercised the lexical firewall twice, which is worth
recording because both were live catches rather than drills:

1. Pernix memories carry `@weight: high`. The evidence-tier ingester rewrites it
   to `@salience` at the boundary, which is exactly the rename §3.2 mandates —
   the firewall failed on real data until the rename was implemented.
2. The word "counterweight" in a source comment tripped the scan. The spec's
   check is a plain substring `grep -rn weight`, so this is correct behaviour,
   not a false positive; the comment was reworded.

### D15 — `ts` is wall-clock at append, recorded in the event
The event's timestamp is written into the chained line, so replay reads the
same value it originally hashed. Nothing downstream calls the clock, which is
why `closure_hash()` is stable across rebuilds.

---

### D19 — v0.3 delta adopted (supersedes D16's "deviation" status)
`docs/spec-v0.3-delta.md` amends the v0.2 spec on the strength of the
pre-registered 6.8 failure: two-coin actor confusion (Δ1), context-grouped
sub-additive composition (Δ2), re-scoped §3.12 accuracy claim + permanent
uniform control (Δ3), the crisp Beta(1,1) prior codified (Δ4, formerly D16),
and dense ranking admitted as an untrusted periphery input by the principal's
explicit selection (Δ5 = R2). The v0.2 conformance harness remains frozen and
green against the v0.3 build; new behaviour is covered under tests/unit/.

## Build-order honesty

§8 requires stage-by-stage work with a commit at each green gate. The trusted
core landed as one body of work rather than as five separately-committed
slices; the gates were then run and committed **in ascending order, each
verified green before the next was run**, and the gate output is pasted into
each commit message. The commit sequence is an accurate record of gate
passage, not a reconstruction of an incremental build. Stated plainly rather
than dressed up.

---

## Open items

1. **High rejection rates thin the accepted sample.** Two near-certain
   mutex-exclusive facts reject ~98% of worlds, leaving ~9 accepted samples out
   of 512, so `p` and `ci` under heavy constraint tension are coarse. `S` should
   arguably adapt upward when `rejection_rate` is high. Flagged, not fixed.
2. **Test 6.8 (the honest test) has not been run.** It needs pre-registered
   margins and a baseline corpus, neither of which exists yet. See the run
   report for exactly what is required.
3. **Stage 5 is not built.** `test_g_changepoint_vs_guard_routing` and
   `test_s_dispersion_power` fail, as expected for a Stage-4 build: the
   curiosity engine's statistics exist and are unit-tested, but nothing yet
   turns them into gate candidates or sets `facts.dispersion_flag`.

### D20 — Stage-5 sweep semantics (v0.4)
Three choices the spec leaves open, made and tested in `periphery/curiosity_engine.py`:
1. **Memoryless detection.** The sweep re-proposes a persisting pattern every
   run rather than deduplicating against repair history — detection stays a
   pure function of the observation log (replay-equivalent), at the cost of
   proposal churn absorbed by gate idempotence.
2. **Cardinality rail.** A covariate whose value count exceeds
   n/(2·min_support) licenses *detection* (flag + open question) but never a
   guard — an m-ary split at that granularity is a lookup table, not a
   condition. This is what keeps bookkeeping keys (`batch`, `t`) from gaming
   the MDL check.
3. **Recurrence discriminator.** §4.4 routes time-clustered surprise to
   changepoint and target-clustered surprise to dispersion, but CUSUM alone
   cannot tell a step from an oscillation. The sweep locates the change at the
   argmax of cumulative deviation and asks whether either remaining half
   changes again: one-way → supersede-with-valid-time; recurring → dispersion
   question (the repair is a condition, not a regime).
Held-out validation for discovered guards (§3.4 step 5) splits observations by
event parity: even indices discover, odd indices validate direction.

### D21 — source retraction is a first-class event, distinct from redaction
The spec gives one purge primitive, `redaction`, keyed on a payload hash.
Payloads are content-addressed and carry no actor, so two sources reporting the
same outcome on the same statement in the same context share one payload:
redacting a liar's hashes destroys the honest reports that agreed with it
(measured: 185 of 600 honest observations). Since multi-source agreement is the
situation the substrate exists to reason about, the documented recovery path was
most destructive exactly where it mattered.

A `retraction` event kind is added, keyed on the **actor**. Exclusion happens at
fold time: the retracted actor's event skeletons stay in the chain forever
(nothing is erased, I3 holds) but contribute no payload, so every downstream
number — including trust, since `_apply_resolution` re-scores on replay —
recomputes as if the source never spoke. Append-only and reversible
(`restore=True`); last write per actor wins. `redact` keeps its content-purging
meaning, which is a real and separate need, and now reports its blast radius via
`redaction_scope` plus a diagnostic when a payload is shared.
Evidence: `bench/CLAIMS_HARDENING.md` Stage 2.

### D22 — operator reliability overrides temper the crisp vote path
v0.3 Δ1 replaced the epistemic Beta with attributed two-coin votes for crisp
facts, which read `actor_confusion` (settlement-moved only). `set_reliability`
writes `actor_reliability`, so after Δ1 the operator's discount lever silently
did nothing to crisp facts — the statement type `docs/use-cases.md` uses it on.
An explicit override now also tempers each of that actor's votes in log-odds
space (`reliability.temper`), the same device Δ2 uses to price correlated votes
sub-additively. Only *explicit* overrides temper: learned reliability already
speaks through the confusion ledger, and folding E[rel] in as well would
double-count the same settlements, so a store with no overrides composes
byte-identically to before. Evidence: `bench/CLAIMS_HARDENING.md` Stage 3.

### D23 — changepoint significance is exact, and the supersede is gated
Supersedes D20.3. CUSUM normalises its alarm by `sqrt(p(1-p))`, a Gaussian
approximation; Bernoulli increments are badly skewed near 0 and 1, so the
recurrence discriminator false-alarmed on 40% of *stationary* p=0.95 segments
against 3% at p=0.5, destroying 64% of genuine 0.95→0.05 breaks — the failure
was worst in the broken-tool regime the feature exists for.

Localization is unchanged (argmax of cumulative deviation, median error 1
observation in 120). The significance decision around it is now
`fisher_exact` — a two-sided hypergeometric p-value in log space, exact at any
base rate — Bonferroni-corrected by the number of split positions searched,
with the same machinery applied inside each segment for recurrence.

`supersede_valid_time` was also the only candidate kind the gate admitted
unconditionally, so periphery false positives became committed history. It now
runs steps 1/5/6: the fact must exist, the proposal must carry a located
`valid_to` (a regime change that cannot say *when* is not one), both regimes
need ≥`GUARD_MIN_SUPPORT` observations, and the level change must clear
`SUPERSEDE_ALPHA` after correction — tighter than the guard's BH α, because a
false supersede rewrites history while a false guard is only a rejected
candidate. The located date is now carried in the body and committed, where
`apply` previously fell back to the sweep's own wall clock.
Evidence: `bench/CLAIMS_HARDENING.md` Stages 4-5.

### D24 — Tarone's Z denominator corrected; instability is detectable without a covariate
Two changes to §4.5 dispersion, one a bug fix and one a gap.

**The statistic.** `tarone_z` multiplied its denominator by `p/(1-p)`, which is
not part of Tarone's Z (the null standard deviation of the chi-square term is
`sqrt(2 Σ nᵢ(nᵢ-1))` and carries no p). Measured false-discovery rate on
covariate splits with no real effect, BH at 0.05: **0.407 at a 5% base rate**,
0.263 at 10%, 0.000 at 90%. An ordinary mostly-failing scraper was handed a
fabricated "works when X" 41% of the time. Corrected, the rate is flat at
3.5-6.8% across p ∈ [0.05, 0.95] and the statistic is *more* powerful on real
structure, so nothing was traded for the calibration.

**The gap.** The flag/question branch required some *recorded covariate* to be
overdispersed, so a stream swinging 85%/35% with nothing useful logged produced
no signal at all — while the time axis had already detected the instability and
the routing discarded it. `suggested_measurement([])` ("log wider: the missing
argument was never captured") was unreachable. `temporal_dispersion` now tests
overdispersion across contiguous time blocks at several scales, corrected for
the scales tried; it needs no covariate because the variance is visible in the
series itself. The routing speaks on either ground, and for the temporal case
the residual partition *is* the time blocks.
Evidence: `bench/CLAIMS_HARDENING.md` Stage 6.

### D25 — v0.5 delta adopted: open-vocabulary categorical facts + distribution surfacing
A third `stmt_type`, `categorical`, whose observations record a *value* over an
open (growing) vocabulary, plus a read-time `distribution()` X-ray for flaky
binary facts. Decisions locked before the build:

- **Unknown mass is Dirichlet-process / CRP with the Pitman–Yor discount pinned
  to `d = 0`.** `P(unknown) = alpha/(N+alpha)` is a function of `N` alone; making
  novelty depend on the distinct-value count is the deferred `d > 0` upgrade. A
  single pre-registered global `alpha = 1.0` is versioned into the predictor
  class `categorical/v1` and rides the snapshot id (I8); the categorical path
  calibrates under its own predictor class, never pooled with the scalar path
  (I9). Per-value intervals are the Beta marginals of the Dirichlet, reusing
  `betamath` verbatim.
- **Per-source trust is a one-vs-rest reduction** over the existing two-coin
  confusion machinery (virtual actor id `catv1:<canon_json([actor, value])>`),
  not a full Dawid–Skene value→value confusion matrix (deferred).
- **Leaf-query only in v1**, one global `alpha`. Joint-multinomial guard tests
  and per-value changepoints are deferred.
- **Strictly additive.** The only schema tightening that widens is the
  `facts.stmt_type` CHECK; a crisp/frequency prediction is byte-identical whether
  or not categorical data shares the store. `fact_category_counts` and the
  nullable `observations.value` fold on the same path and enter `COUNT_COLUMNS`
  (I11) and `_HASH_QUERIES`.

Evidence: `docs/spec-v0.5-delta.md`; `tests/unit/test_categorical_c1.py`..`c4.py`,
`tests/unit/test_distribution_surfacing.py`.

### D26 — quotas are per-epoch, with the epoch derived from the event timestamp
§3.12's `obs_quota_per_epoch` was enforced with `epoch` pinned to 0 everywhere —
a lifetime cap that never reset. The epoch is now `apply.epoch_of(ts) =
ts // QUOTA_EPOCH_MS` (one day); observe()/assert_() fix the event ts up front so
the boundary check and the fold-time bump agree on the bucket. Quotas remain
deployment configuration outside the closure hash (`quota_usage` is absent from
`_HASH_QUERIES`), so this moves no replay number, and replay reproduces every
bucket because the ts is in the ledger. A live burst (default wall-clock ts)
stays inside one epoch, so the flooding bound is unchanged.
Evidence: `tests/unit/test_quota_epoch_m2.py`.

### D27 — authorization is opt-in; attribution labels are advisory by default
`actor`/`authority`/`source` are attribution labels, not authenticated
identities — CANDOR's trust boundary is the process. `set_authz(policy)`
installs a `(principal, op) -> bool` gate consulted before the privileged writes
(pin, redact, retract_source, register_oracle, set_reliability), raising
`Unauthorized` before any ledger append. The policy is runtime configuration: it
never enters the closure hash and replay never re-checks it (every event already
in the chain was admitted under whatever policy was active when written).
Relatedly, the hash chain is a *consistency* mechanism, not tamper-evidence — a
writer who can edit a segment can recompute it — so tamper-evidence against that
threat is left to an external anchor of `ledger_head()`.
Evidence: `SECURITY.md`; `tests/unit/test_authz_m3.py`.

### D28 — the sweep searches synthesized frames, not just recorded context
§4.5 scopes the covariate search to recorded `obs_context` keys. v0.6 Δ10
extends it: hour/dow from the event ts, the fact's previous outcome, and
pairwise interactions of recorded keys are synthesized per fact and tested by
the identical Tarone→BH→MDL→held-out flow. Two conservatisms are ours, not the
spec's: recorded keys outrank derived ones at winner selection, and
`derived:prev` is disqualified when a one-way changepoint exists or when either
prev-conditioned subseries still carries temporal structure (self-lag shadows a
date or an unlogged block variable otherwise). Breadth stays recorded-only.
All of it is a pure per-fact function of the log, so H6b/I3/I8 hold.
Evidence: `docs/spec-v0.6-delta.md`; `tests/unit/test_derived_keys.py`.

### D29 — §3.4 demotion is finally load-bearing, with evidence measured in nats
The spec promised "demotion runs the same path backward with a strictly higher
bar" and the fold always knew the `demotion` event — but nothing ever appended
one. v0.6 Δ11 activates it via the prospective audit in `run_gate`. One
interpretive choice: the hysteresis comparison uses signed binomial
log-likelihood-vs-chance ("nats", `gate.direction_evidence`), because the naive
count-odds ratio saturates with sample size and makes any hysteresis bar
unreachable. Second choice: staleness (direction stops beating chance on twice
the entry evidence) demotes alongside reversal — a rent check, not just a
falsification check. Re-entry is judged on post-demotion evidence only.
Evidence: `tests/unit/test_guard_prospective_audit.py`.

### D30 — committed conjectures transfer the analog's probability, not similarity
§4.3 defines conjectures as read-only proposals. v0.6 Δ12 adds an opt-in commit
path through the claims machinery. The deviation-worthy decision: a conjecture
claim's `predicted_p` is `predict(via).p` — the analog's earned number moved
across the soft edge — never the cosine similarity, which is a licence, not a
probability. The engine calibrates under its own `conjecture/v1` class (I9) and
a true settlement implements the postulate through the gate (I10).
Evidence: `tests/unit/test_conjecture_claims.py`.

### D31 — `do:` is vocabulary for intervention, deliberately not a causal model
v0.6 Δ13 reserves the `do:` context prefix. We chose surfacing over inference:
mixed-regime predictions still report the pooled marginal (predictions are
unconditional by architecture) with a loud `regime_mixed` caveat, and a guard
on a `do:` key is labeled regime dependence rather than becoming a conditional
prediction. Predicting what an intervention will change before
post-intervention data exists is out of scope by design — the claims-suite
battery asserts that boundary so it stays a measured fact about this system.
Evidence: `tests/unit/test_do_semantics.py`; `tests/claims/test_axiom_battery.py`.

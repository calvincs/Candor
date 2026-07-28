# CANDOR — Calibrated Agent Memory Substrate

**Working spec, v0.2** · supersedes v0.1 · codename is a placeholder, rename freely.

A memory layer for AI agents in which retrieval, admission, prediction, and settlement
share one provenance chain. Facts can be crystalline or provisional; weights move only
on observed outcomes; every number is auditable back to the events that produced it;
instability in a "fact" is treated as evidence of a missing variable rather than as
uncertainty about truth.

Inherits functional ideas from HyperKB (plain-text evidence tier, hybrid search,
token-budgeted retrieval) and Provenas (test-before-admit gate, derivations, sandboxed
tool synthesis). Shares no code with either. Stands alone.

**v0.2 delta:** closes the observation trust hole, defines two-channel semantics and
their composition, adds symbol identity (registry + aliases), makes time a structural
covariate with valid-time regimes, structures `context_sig`, fixes API type-honesty,
adds integrity constraints, specifies the ledger's physical form, and replaces §6 with
a full conformance test harness. Appendix A maps every change to the review item that
motivated it.

---

## 0. Design invariants

These are non-negotiable. Every component below exists to serve one of them. If an
implementation choice violates one, the choice is wrong.

| # | Invariant | Why |
|---|---|---|
| I1 | **The ledger is the only primary artifact.** Physically: append-only segments plus a content-addressed payload store. Every other store — including SQLite — is a materialized view, droppable and rebuildable. | Makes counts auditable, calibration retroactive, retraction tractable, and corruption recoverable. |
| I2 | **Retrieval never moves a weight.** Enforced by type *and* by stream: retrieval events live outside the primary chain and the count updater has no import path from them. | Otherwise the graph manufactures its own consensus: rich-get-richer, confident and self-generated. |
| I3 | **Nothing is mutated.** Change = append + recompute. | "Crystalline" becomes a permission set, not a storage location. |
| I4 | **A derivation containing one soft edge is not a proof.** Distinct return type. | Prevents callers from trusting analogy as entailment. |
| I5 | **Zero is reserved for refutation.** Unobserved gets ε. | A hard zero is an absorbing state; unobserved things must be able to come back. |
| I6 | **No claim without a constructible verifier.** | Unsettleable statements are prose, not claims. Keeps calibration meaningful. |
| I7 | **Epistemic and aleatoric uncertainty never share a number.** Statement type (§4.2) determines which channel an event moves; composition is by two-loop sampling (§3.9). | They compose differently through inference. |
| I8 | **Predicted probability is recorded at prediction time with a model snapshot id.** Snapshot ≔ {ledger head hash, engine version, calibration map hash}. | Recomputing it later — including via a refit isotonic map — contaminates calibration silently. |
| I9 | **Calibration is partitioned by (frame, settlement, predictor class), never pooled.** | A reliability diagram over mixed reference classes — or mixed predictors — looks fine and means nothing. |
| I10 | **The LLM lives only at the candidate boundary.** It proposes; it never writes an edge or a weight. | Confines the untrusted structural component to one auditable surface. |
| I11 | **Stored counts are integers, keyed by (actor, channel). Every real-valued number is a read-time composition** — reliability discount, alias union, isotonic map — never storage. | Learning must not corrupt the audit trail. A discounted count can be recomputed forever; a mutated one cannot. |

### Non-goals

- Not a vector store, not a replacement for one. No text embeddings anywhere in the core.
- Not a general-purpose graph database.
- Not a causal inference engine. Discovered conditions are **conditioning, not causal** —
  labeled as such, and never sold as intervention-safe.
- No distributed consensus in v1. Single writer to the gate, single sequencer for the ledger.
- No defense against long-horizon reputation farming in v1. Quotas and frame-partitioned
  actor reliability bound the damage; a full defense is documented as out of scope (§3.12).
- v1 outcomes are **binary**. Scalar outcomes need predictive densities and CRPS-style
  scoring; explicitly deferred (§9).

### Semantic fork, decided

Probability semantics via **weighted model counting**, not fuzzy t-norms. Truth degrees
under Łukasiewicz/PSL give convex MAP inference and scale better, but they are not
probabilities and cannot be calibrated — which forfeits I8/I9 and the entire point.
WMC is #P-hard; §7 defines the budget and degradation path. The fuzzy fallback is
**refused**, not deferred: `recall` is already the sanctioned uncalibrated read path,
and a second one returning number-shaped non-probabilities is the confidently-meaningless
failure this design exists to prevent. Channel composition through WMC is settled by the
two-loop sampling scheme in §3.9.

---

## 1. Architecture

```
                      ┌─────────────────────────────────────────────┐
   inputs ───────────►│ LEDGER: append-only JSONL segments, hash    │
   (docs, tool output,│ chain over payload COMMITMENTS; payloads in │
    agent assertions, │ a content-addressed store (redactable).     │
    resolved outcomes)│ SQLite `events` table = derived index.      │
                      └───────────────┬─────────────────────────────┘
   retrieval side stream              │ replay = recompute
   (outside the chain, I2)            │
                     ┌────────────────┴────────────────┐
                     ▼                                 ▼
       ┌───────────────────────┐        ┌──────────────────────────────┐
       │ EVIDENCE TIER         │        │ COMMITTED TIER               │
       │ prose .md, append-only│        │ predicate registry, facts,   │
       │ stable entry ids,     │        │ rules, constraints, aliases, │
       │ @salience ranking,    │        │ pins, proofs, actor-keyed    │
       │ decayed, retrieved    │        │ counts, closure              │
       │ = covariate reservoir │        │ = crystalline + provisional  │
       └──────────┬────────────┘        └──────────┬───────────────────┘
                  │      ┌───────────────────┐     │
                  └─────►│ PROMOTION:        │─────┘
                         │ extract→candidate │
                         │ →GATE (canonical- │
                         │ ize, test, admit) │
                         └───────────────────┘
   read paths:  RECALL (prose) · DERIVE (exact; three-valued return) ·
                CONJECTURE (soft) · PREDICT (two-loop sampled WMC)
```

### Trusted core / untrusted periphery

LCF-style separation. The trusted core is small, dependency-light, and independently
testable. Everything else may be wrong without corrupting the store.

**Trusted (must be provably correct):**
ledger append + hash chain + redaction handler · canonicalizer (units, arg normalization) ·
closure engine · constraint checker · proof checker kernel · count updater ·
actor-reliability scorer · pin enforcement · calibration bucketer

**Untrusted (fallible, always gated, always attributed):**
LLM extractor · rule proposer · alias proposer · verifier/tool synthesizer ·
soft-edge neighborhoods · guard discovery · changepoint detector · summarizers

The canonicalizer is trusted *harness* over untrusted *content*: normalization rules are
deterministic and registry-driven; the proposal being normalized is not.

---

## 2. Data model

**Physical layout.** The ledger is flat append-only JSONL **segments**, hash-chained,
with payloads stored separately in a content-addressed store (`payloads/<hash>.json`).
SQLite holds the committed tier and a derived **index** of the ledger; deleting the
SQLite file loses nothing (I1). Evidence tier is plain `.md` files with stable entry ids.
Python 3.11+, stdlib-first.

```sql
-- ── LEDGER INDEX (derived; the JSONL segments are the primary artifact) ────
events(
  id            INTEGER PRIMARY KEY,
  ts            INTEGER NOT NULL,
  kind          TEXT NOT NULL CHECK(kind IN
                  ('assertion','observation','supersede','admission','demotion',
                   'pin','claim','resolution','alias','redaction','checkpoint')),
  actor_id      INTEGER NOT NULL REFERENCES actors(id),
  payload_hash  TEXT NOT NULL,          -- commitment; payload lives in CAS, deletable
  source_ref    TEXT,                   -- (entry_id, content_hash, offset) | url | tool run id
  context_sig   TEXT,                   -- derived: hash(canonical(obs_context)); see §4.6
  prev_hash     TEXT NOT NULL,
  hash          TEXT NOT NULL           -- over (ts,kind,actor,payload_hash,prev_hash,...)
);
-- 'retrieval' is GONE from the chain. Retrieval logging is a side stream:
retrieval_log(ts, actor_id, query, spans_json);   -- no chain, no downstream writes (I2)
-- Only 'observation' and 'resolution' may reach the count updater. Enforced in code
-- by separate writer types with no shared interface, verified by provenance scan (§6.2).

-- ── ACTORS & RELIABILITY ──────────────────────────────────────────────────
actors(
  id, name, class CHECK(class IN ('human','verifier','tool','agent')),
  obs_quota_per_epoch INTEGER,          -- flooding bound (§3.12)
  cand_quota_per_epoch INTEGER          -- gate-flooding bound (§3.4)
);
actor_reliability(
  actor_id REFERENCES actors(id),
  frame    CHECK(frame IN ('internal','external')),
  rel_a REAL, rel_b REAL,               -- Beta; moves ONLY on trusted-path scoring (§3.12)
  PRIMARY KEY(actor_id, frame)
);

-- ── IDENTITY & SCHEMA ─────────────────────────────────────────────────────
predicates(
  pred TEXT PRIMARY KEY,
  arity INTEGER NOT NULL,
  arg_types_json TEXT NOT NULL,         -- typed args make guards & arithmetic possible
  canonical_units_json TEXT,            -- per-arg canonical unit; gate normalizes to it
  admitted_at, admitted_by_event
);
aliases(                                -- materialized view over 'alias' events
  canonical TEXT, alias TEXT,
  basis CHECK(basis IN ('behavioral','definitional','pinned')),
  admitted_at, admitted_by_event
);
-- Counts are NEVER merged on alias. Union happens at read time through the alias
-- closure (I11), so a bad merge is reversible by supersede and audit stays intact.

-- ── CANDIDATES (never facts) ──────────────────────────────────────────────
candidates(
  id, event_id,
  kind CHECK(kind IN ('fact','rule','guard','verifier','symbol','alias','constraint')),
  body TEXT,                            -- proposed structure
  span_ref TEXT,                        -- (entry_id, content_hash, offset) into evidence
  proposer INTEGER REFERENCES actors(id),
  status CHECK(status IN ('pending','admitted','rejected','superseded')),
  gate_run_id
);

-- ── COMMITTED TIER ────────────────────────────────────────────────────────
facts(
  id, pred REFERENCES predicates(pred), args_json,   -- args unit-canonical post-gate
  stmt_type CHECK(stmt_type IN ('crisp','frequency')),   -- disambiguates channels (§4.2)
  kind CHECK(kind IN ('exact','soft','definitional')),   -- admission basis
  sim REAL,                             -- soft basis only: similarity license
  epi_a REAL, epi_b REAL,               -- CACHE: composed at read (fact_counts × reliability)
  alea_n INTEGER, alea_k INTEGER,       -- CACHE: pooled view of raw actor-keyed counts
  structural CHECK(structural IN ('candidate','admitted','pinned')),
  numeric    CHECK(numeric    IN ('accumulating','frozen')),
  breadth_class CHECK(breadth_class IN ('narrow','moderate','broad')),
  dispersion_flag INTEGER DEFAULT 0,    -- latent variable suspected (§4.5)
  valid_from INTEGER, valid_to INTEGER, -- regime bounds; NULL valid_to = current (§4.5)
  admitted_at, admitted_by_event
);
-- definitional facts (arithmetic, declared type membership) carry NO weights and are
-- ineligible for blame. They are outside the weight system, not high numbers within it.
-- Definitional test: derivable by the trusted core alone, with no reference to counts
-- and no world input. Closure membership FAILS this test and doesn't need to pass it:
-- exact-derived facts are already blame-ineligible (§4.4).

fact_counts(                            -- the RAW counts. Integers. The audit trail.
  fact_id REFERENCES facts(id),
  actor_id REFERENCES actors(id),
  channel CHECK(channel IN ('epi','alea')),
  n INTEGER NOT NULL, k INTEGER NOT NULL,
  PRIMARY KEY(fact_id, actor_id, channel)
);
-- facts.epi_*/alea_* are recomputable caches:
--   effective(channel) = Σ_actor  count(actor) × E[reliability(actor, frame)]
-- Fractional values appear only in the composed view, never in storage (I11).
-- Retroactive exclusion of a compromised source = recompute. Nothing to unwind.

rules(
  id, head, body_json,                  -- guards allowed in body: binned comparisons,
                                        -- monotone bin constraints (§4.5); no function
                                        -- symbols in heads (stays inside Datalog)
  specificity INTEGER,                  -- for conflict resolution (§4.5)
  parent_rule_id,                       -- set when this is a discovered refinement
  w_a REAL, w_b REAL,                   -- CACHE like facts; raw in rule_counts (same shape)
  structural, numeric,
  gate_run_id, admitted_at
);
rule_counts(rule_id, actor_id, n INTEGER, k INTEGER, PRIMARY KEY(rule_id, actor_id));

constraints(                            -- first-class, gate-admitted (§3.4 step 7)
  id, kind CHECK(kind IN ('mutex','functional')),
  body_json,                            -- mutex: exclusive predicate/value sets
                                        -- functional: pred is single-valued in arg positions
  structural, admitted_at, gate_run_id
);

pins(id, target_kind, target_id, polarity CHECK(polarity IN ('+','-')),
     reason, authority, created_at);
-- polarity '-' is the ONLY hard zero in the system. (I5)
-- Observations contradicting a '-' pin are absorbed but counted; past the surprisal
-- threshold they open a pin_tension question (§3.10). The pin still wins. A human
-- gets paged.

-- ── CLAIMS & CALIBRATION ──────────────────────────────────────────────────
claims(
  id, stmt_json,
  frame      CHECK(frame IN ('internal','external')),        -- what it's about
  settlement CHECK(settlement IN                             -- how it closes
                ('entailed','tool_decidable','observation_pending','unsettleable')),
  -- frame and settlement are ORTHOGONAL. An internal claim about runtime behavior
  -- still needs execution to settle.
  verifier_id, due_ts,
  predicted_p REAL, predicted_ci_lo REAL, predicted_ci_hi REAL,
  model_snapshot TEXT NOT NULL,          -- {ledger_head_hash, engine_version,
                                         --  calib_map_hash}  (I8)
  certainty_class CHECK(certainty_class IN ('certain','high','estimated','unlicensed')),
  resolved_ts, outcome, surprisal REAL,
  CHECK (settlement = 'unsettleable' OR verifier_id IS NOT NULL)   -- (I6)
);
-- resolution event payloads additionally record: verifier_code_hash, env_hash, and the
-- full sensitivity vector of the derivation (§4.4). Settlements are re-audit-able.

proof_steps(claim_id, step_no, rule_id, fact_id, edge_kind, sensitivity REAL);

oracles(id, kind CHECK(kind IN
          ('deterministic_total','deterministic_partial','stochastic')),
        impl_ref, code_hash, env_hash, n_trials, n_correct, validated_at);
-- only deterministic_total yields certainty_class='certain'. Everything else carries
-- the oracle's own reliability into the claim. deterministic_* kinds must reproduce
-- bit-for-bit on re-run in the pinned env, or are demoted to 'stochastic' (§6.6).

calibration(frame, settlement, predictor_class, bucket,
            n, mean_p, observed_freq, updated_at);
-- partitioned (I9). predictor_class = coarse (engine family, model family), not
-- snapshot — snapshots are recorded per-claim for audit, pooled by class for power.
-- Alerting requires n ≥ min_n (default 50) per bucket.

-- ── LEARNING & CURIOSITY ──────────────────────────────────────────────────
signatures(sym, sig_sparse BLOB, computed_at);   -- behavioral, view over closure
soft_edges(from_sym, to_sym, sim REAL, basis TEXT, computed_at);

obs_context(                            -- structured ambient state (§4.6)
  event_id REFERENCES events(id),
  key TEXT, value TEXT,
  PRIMARY KEY(event_id, key)
);
-- context_sig = hash(canonical_serialization(obs_context rows)). The hash is for fast
-- grouping; covariate search and per-key breadth operate on the COMPONENTS.

open_questions(
  id, kind CHECK(kind IN ('dispersion','pin_tension')),
  target_kind, target_id,
  residual_partition TEXT,               -- the observations that disagree
  dispersion_stat REAL,
  ruled_out_json TEXT,                   -- covariates already tested and cleared
  suggested_measurement TEXT,            -- experimental design (§4.5)
  status CHECK(status IN ('open','explained','abandoned')),
  explained_by_guard_id
);

invariants(id, family, target_scope, fail_policy CHECK(fail_policy IN
           ('fail_stop','alert_only')), last_run, status, failure_ref);
eval_queue(target_kind, target_id, dependents INTEGER, sensitivity REAL,
           cost REAL, score REAL);      -- score = dependents × sensitivity ÷ cost
-- cost sources: static defaults per settlement type (entailed ≈ 0; tool_decidable =
-- sandbox estimate; observation_pending = elicitation constant), refined from history.
```

---

## 3. Components

### 3.1 Ledger (trusted)

Append-only JSONL segments, hash-chained over **payload commitments**; payloads live in
the content-addressed store. Two input writer types remain strictly separated:
`AssertionWriter`, `ObservationWriter`. `RetrievalLog` writes only to the side stream
and has no reference to the count updater and cannot obtain one.

**Durability.** Admission, resolution, pin, supersede, alias, and redaction events fsync
immediately; observation events may batch-fsync (configurable, default every 32 events
or 500 ms). Torn-write recovery: on open, truncate the tail segment to the last line
whose hash verifies; anything after is lost, never silently corrupted.

**Redaction.** A `redaction` event names a payload hash; the payload file is deleted;
the chain — which covers only commitments — still verifies. Redaction implies
**exclusion**: replay recomputes all downstream state *without* the redacted content,
which is exactly the desired semantics for purging a bad or sensitive source. Replay
determinism (§6.2) is defined against post-redaction state. The event skeleton
(ts, kind, actor, hashes) remains forever; the content does not.

**Retraction is not implemented.** Supersede appends a `supersede` event and the closure
is recomputed from the log. At ~50k facts this is seconds; correct by construction
rather than correct-if-the-TMS-is-right. Keep JTMS/DRed in reserve for §7 thresholds.
`checkpoint` events record the ledger head hash they summarize; rebuild-from-checkpoint
must be spot-verifiable against full replay (§6.6).

### 3.2 Evidence tier

Plain markdown, dotted-namespace filenames, hybrid retrieval (ripgrep exact/regex +
FTS5 BM25), recency decay, token-budgeted context packing via greedy knapsack.
Metadata: `@type`, `@status`, `@salience`, `@tags`.

**`@weight` is renamed `@salience`.** It is a retrieval-ranking input and nothing else.
The firewall against I2 leakage is now lexical: nothing named "weight" exists outside
the committed tier, and `grep -rn weight` outside it is an audit (§6.2).

**Stable identity.** Every entry gets an `entry_id` at append plus a `content_hash`.
Span references everywhere in the system are `(entry_id, content_hash, offset)` —
file renames and reorganization cannot orphan provenance.

Its second and equally important job: **covariate reservoir**. It retains incidental
ambient detail a schema would discard. When a latent variable is eventually identified,
the material that explains it is frequently already sitting here unstructured. This is
the load-bearing reason for the two-tier split, not just a stylistic preference.

### 3.3 Extractor (untrusted)

Evidence → `candidates`, each carrying `span_ref`. Emits candidates only. Cannot write
to `facts`, `rules`, or any weight column. (I10)

**Registry linkage.** Extracted facts must reference a registered predicate, or the
extractor emits a `symbol` candidate proposing the new predicate (arity, arg types,
units) alongside the fact candidate that needs it. Suspected co-reference between
symbols is proposed as an `alias` candidate — never assumed.

### 3.4 Gate (trusted harness, untrusted content)

Single serialization point for all structural change. Admission requires:

1. Syntactic/AST validation against the predicate registry (arity, arg types).
2. **Canonicalization**: units normalized per registry (`212F → 373.15K`), argument
   normal forms applied. Deterministic, registry-driven, trusted.
3. Sandboxed execution for synthesized verifiers and tools.
4. **Pinned regression cases pass.** A pin can veto any candidate.
5. Held-out evidence check for rules and guards.
6. MDL improvement for guards (§4.5).
7. Contradiction check, now well-defined: `closure ∪ candidate` must violate no
   admitted **constraint** (mutex, functional). Constraints themselves are candidates
   admitted through steps 1–5.

Emits an `admission` event with a `gate_run_id`. Rejections are recorded, not discarded —
they are training signal and they prevent re-proposal churn.

**Quotas.** Per-proposer candidate quotas (`actors.cand_quota_per_epoch`) bound gate
flooding by a chatty or adversarial proposer. Quota exhaustion is an event, visible in
`health()`.

**Alias admission** requires one of: behavioral-signature similarity above threshold
*with* zero constraint conflicts between the merged extensions; definitional identity
(pure unit/notation conversion); or a human pin. Union-at-read (I11) makes a wrong
alias reversible by supersede.

**Demotion runs the same path backward with a strictly higher evidence threshold than
admission** — concretely: posterior odds against the structure must exceed the admission
odds by a configured hysteresis factor (default 3×). Without it a noisy observation run
evicts a good rule and the graph oscillates.

### 3.5 Closure engine (trusted)

Stratified-negation Datalog with comparison guards over **binned values and monotone
bin constraints** (§4.5) — no function symbols in rule heads, so the engine stays
Datalog. Materializes exact closure; enforces admitted constraints during
materialization. Soft edges participate only in `conjecture` mode and are tagged through
every step, so the planner can refuse to label the result a proof. (I4)

### 3.6 Proof checker kernel (trusted, tiny, isolated)

Independent of the search that produced the derivation. Verifies that each emitted
conclusion follows from its cited premises. Checking is polynomially cheap relative to
finding, so this runs on every emitted proof.

This is the only component whose correctness is provable in the strict sense, and the
only eval in the system that catches bugs in the engine itself.

### 3.7 Count updater (trusted)

Counts, never deltas — and now **integer counts keyed by (fact, actor, channel)** in
`fact_counts` (I11). Consumes `observation` and `resolution` events exclusively. The
per-fact `epi_*`/`alea_*` columns are lazily recomputed caches:

```
effective(fact, channel) = Σ_actor  raw(fact, actor, channel) × E[rel(actor, frame)]
```

- Which channel an observation moves is determined by the fact's `stmt_type` (§4.2).
- Unobserved: ε via Dirichlet pseudocounts / reserved unseen-mass bucket. (I5)
- Frozen (`numeric='frozen'`): updater is a no-op. This is how a structurally
  permanent-and-mobile rule differs from a definitional truth.
- **Committed counts never decay.** Non-stationarity is repaired by regime detection →
  supersede-with-valid-time (§4.5), never by a decay hyperparameter. A fact confirmed
  500× in 2019 and contradicted 10× this month is a *changepoint*, and the machinery
  must say so out loud rather than quietly forgetting.

### 3.8 Claim registry & settlement triage

Every claim is triaged on creation:

| Settlement | Meaning | Resolution |
|---|---|---|
| `entailed` | follows from committed facts by exact derivation | settles now, no world needed |
| `tool_decidable` | a verifier can be synthesized | sandboxed run, oracle reliability applies |
| `observation_pending` | needs the world | due date, waits |
| `unsettleable` | no procedure constructible | **refused entry**; stays prose in evidence tier |

`entailed` claims are the calibration engine: mask a known fact plus everything derivable
from it, recompute the closure, ask the soft path to predict what was removed, compare
against what the exact path proves. Leak-free ablation is normally the hardest part of
knowledge-graph evaluation; replay-from-log makes it exact and cheap. **This calibrates
the fuzzy machinery today, over thousands of cases, before any external outcome resolves.**

Counterweight: self-consistency is not truth. If a rule is wrong, the closure is
consistently, provably, auditably wrong and every entailed eval passes. Maintain a
configured floor on the external-settled ratio; alert when it is breached.

Resolutions record `verifier_code_hash` and `env_hash` in the event payload — a patched
verifier can never make an old settlement unauditable.

### 3.9 Prediction engine

Weighted model counting over the proof DNF, wrapped in an **epistemic outer loop** that
settles channel composition (closes v0.1 open decision 3):

```
for s in 1..S:                                   # epistemic samples
    for each fact f in the support:
        v_f ~ Beta(epi_f)                        # validity posterior
        t_f ~ Bernoulli(v_f)                     # crisp truth / class validity, THIS world
        θ_f ~ Beta(alea_k+α, alea_n−alea_k+β)    # frequency facts only
        lit_weight(f) = t_f                      if stmt_type = crisp
                      = t_f × θ_f                if stmt_type = frequency
    if world violates an admitted constraint: reject sample, count rejection
    p_s = WMC(proof DNF | lit_weights)           # inclusion–exclusion / d-DNNF / top-k
report:
    p        = mean(p_s over accepted samples)
    ci       = quantiles of p_s                  # epistemic interval
    channels = {epistemic: spread across s; aleatoric: within-world mass from θ}
    rejection_rate                               # constraint tension diagnostic
```

Facts are sampled **once per world**, which is precisely why epistemic uncertainty
correlates across queries sharing a fact and aleatoric does not — the channels compose
differently by construction, not by bookkeeping (I7). Crisp-only proofs yield a
degenerate aleatoric channel (each `p_s ∈ {0,1}`); the mean is purely epistemic and is
reported as such. Worlds violating admitted constraints get zero mass; the count
renormalizes over consistent worlds. A high rejection rate means the epistemic
posteriors are in tension with the constraints — a health signal, surfaced.

Proofs share facts, so probabilities cannot be summed — inclusion–exclusion below the
proof budget, knowledge compilation (d-DNNF) above it, top-k proof bounds when
compilation exceeds budget. **Budget degrades S first**, then exact → compile → top-k;
degrade, never hang. Recalibration applied at **read time** via isotonic map; stored
counts are never mutated by calibration; the map's hash is part of every snapshot (I8).

Always available alongside a prediction:
- **Sensitivity**: which fact, if flipped, changes the conclusion.
- **MPE**: most probable explanation.
- **Caveat set**: inherited under-specification flags (§4.7), plus `shared_provenance`
  when ≥2 premises trace to the same source document — flagged, not modeled; the
  independence assumption is documented where the numbers are consumed.

### 3.10 Curiosity engine

Owns `open_questions` (both kinds), dispersion testing, **changepoint detection on the
time axis**, guard proposal, retroactive re-testing, and breadth accounting. Detail in
§4.5–4.7. Pin-tension questions open when contradicting observations against a `-` pin
exceed the surprisal router's threshold; the pin holds, the human is paged.

### 3.11 Eval scheduler

Single priority queue over everything unresolved — claims, anomalies, invariant runs,
low-breadth facts, pin tensions. Score = downstream dependents in closure × sensitivity
÷ resolution cost. All three terms computable from stored state (cost sources in §2
schema comment). This is what lets an agent decide which unknown is worth resolving
next instead of gathering evidence at random.

### 3.12 Actor reliability (trusted scorer, new)

Observations reach counts without a gate — the world must be allowed to surprise you —
so trust is handled by **attribution plus discount**, not by blocking:

- Every observation is keyed by actor in `fact_counts` (I11). Retroactive exclusion of
  a compromised source is a pure recompute.
- Reliability moves **only through the trusted path**: when a claim settles via a
  `deterministic_total` oracle, every prior observation event on that statement is
  scored against the settled outcome — agreement increments `rel_a`, disagreement
  `rel_b`, per `(actor, frame)`.
- Read-time composition discounts each actor's counts by `E[rel]`. A hallucinating
  agent's influence decays as its disagreements with settled reality accumulate, and
  its past damage is reversible because raw counts were never pooled.
- `obs_quota_per_epoch` bounds flooding; `cand_quota_per_epoch` bounds gate spam.

**Documented limitation:** an actor can farm reliability by observing statements that
settle trivially, then spend it on lies elsewhere. Frame-partitioned reliability and
quotas raise the cost; a full defense (per-predicate reliability, stake-weighted
scoring) is out of scope for v1 and listed in §9.

### 3.13 Identity & schema (new)

The **predicate registry** is the schema, and it is gate-admitted like everything else —
including its bootstrap (§8 seed path). Typed, unit-canonical arguments are what make
comparison guards, arithmetic bins, and the canonicalization pass possible; the
`boils(water, 212F)` vs `boiling_point(H2O, 100C)` bug dies at admission, not at query
time.

**Aliases** are events with a materialized view; resolution to canonical symbols happens
in the read path, and behavioral signatures are computed over the alias closure so
merged symbols share a neighborhood.

**Cold start, stated honestly:** a new symbol fires no rules, so its behavioral
signature is empty and the conjecture engine cannot see it until signature support
crosses a floor. That is correct behavior, not a bug — analogy from nothing is exactly
the license this system refuses to grant. New domains bootstrap through evidence
accumulation and exact admission, not through soft edges.

---

## 4. Core logic

### 4.1 Lifecycle

```
evidence entry ──► candidate ──► [GATE] ──► admitted fact ──► pinned
                                    │        (kind: exact | soft basis)   │
                                    └────────── demotion (higher bar) ◄───┘
```

`kind` records the **admission basis** (how it earned entry); `structural` records
removability. They are orthogonal (§4.2) — the v0.1 lifecycle line that read
"soft fact → admitted fact" conflated them and is retired. A fact may hold **both**
`exact` and `soft` derivations simultaneously. When the two paths independently reach
the same conclusion, that agreement is free calibration data.

### 4.2 Four orthogonal axes of "crystalline"

Conflating these is the modeling error the whole design exists to avoid.

| Axis | Values | Question |
|---|---|---|
| structural | candidate / admitted / pinned | can this be removed? |
| numeric | accumulating / frozen | can its counts move? |
| kind | exact / soft / definitional | proof, conjecture, or outside the system? |
| stmt_type | crisp / frequency | what do the channels mean? |

**`stmt_type` disambiguates the two channels** (settles the v0.1 ambiguity):

| stmt_type | epi Beta means | alea counts mean | observation event moves |
|---|---|---|---|
| `crisp` | p(statement is true) | unused (NULL) | **epi** |
| `frequency` | p(reference class is valid as stated) | outcome rate among trials | **alea** |

For frequency facts, epi moves only through *structural* events — resolutions of
validity claims, constraint violations, supersede/alias — never through trials. This is
the "I'm 60% sure this fact is true" vs "this outcome happens 60% of the time" split,
now with an explicit event→column mapping instead of a shared number (I7).

They combine independently. A rule can be structurally permanent and numerically mobile:
it stays forever while its reliability floats with observation. A fact can be currently
certain and structurally provisional. **Crystalline is a permission set, not a place.**

### 4.3 Soft unification

Soft edges are generated from **behavioral similarity**, not text embeddings. A symbol's
signature is the set of rules it fires in (with argument positions) plus its
co-derivation set — computable directly from the materialized closure over the **alias
closure**, no embedding model, no network, fully inspectable, and a far better licence
for substitution than "these words look alike."

Sparse vector over `(rule_id, arg_position)`; cosine or Jaccard for neighborhoods;
dimensionality stays in the thousands because it is bounded by rule count. LSH only if
that ever stops being true. New symbols have empty signatures until support crosses a
floor (§3.13) — the conjecture engine is blind there by design.

Soft edges are a **hypothesis generator, never an admission mechanism.** Neighborhoods
propose; observation tests; the gate promotes survivors to exact. Same
propose/verify pattern as the LLM boundary, one level down.

### 4.4 Surprisal router

On every resolution, compute −log p against the recorded `predicted_p`. Route on the
**percentile of that model's own historical surprisal distribution**, not an absolute
threshold — this removes a hyperparameter and adapts as the model improves.

```
low surprisal            → count update only
high, isolated           → local reweight, flag for review
high, clustered on one target → dispersion test (§4.5)
high, clustered on time axis  → changepoint test (§4.5)
high, structural         → soft-edge neighborhood generates candidate rules → gate
```

**Credit assignment.** Blame is apportioned only across components the derivation
actually depended on, ranked by the sensitivity already computed for that proof.
The count increment goes to the **argmax-sensitivity component only** — integer blame,
auditable counts (I11) — while the **full sensitivity vector is logged in the resolution
payload**, so fractional attribution schemes remain recomputable later without ever
having corrupted storage. Ineligible for blame: `definitional` facts, `exact`-derived
facts, `frozen` targets. This forces blame onto rules and soft edges — where learning
belongs. Without the restriction, a bad rule slowly corrodes the truth values of good
facts.

### 4.5 Latent variable detection → guard synthesis

The insight this implements: **an unstable fact was never probabilistic. It is a function
with an unrecorded argument, and the observed variance is the shadow of the missing one.**
Water boils at 212°F holds until it doesn't; the repair is not a lower probability, it is
a discovered condition.

**Detection is statistical, not heuristic.** If outcomes were truly i.i.d. Bernoulli,
counts follow a binomial. Overdispersion is evidence of a latent conditioning variable.

```
group observations for target by batch/context
compute Tarone's Z (or beta-binomial vs binomial likelihood ratio)
if overdispersed beyond threshold:
    set facts.dispersion_flag = 1
    open_question(target, residual_partition, dispersion_stat)
    → covariate search
```

**Time is always a candidate covariate — with its own machinery.** Wall-clock is
structurally included in every dispersion test, but tested with changepoint detection
(CUSUM as the cheap first pass; BOCPD where a hazard prior is worth configuring), not
partition-by-bins, because the repair differs:

- **Confirmed changepoint → supersede-with-valid-time.** The old regime keeps its counts
  and its `valid_from/valid_to`; the successor starts fresh. Prediction defaults to the
  current regime; historical queries address prior regimes by valid time. This is cheap
  bitemporality: transaction time from the ledger, valid time on the fact. "X is CEO of
  Y" is a step function, not an overdispersed coin.
- **Confirmed non-temporal covariate → guard**, as below.
- Both repairs are proposed as candidates and pass **through the gate** like everything
  else.

**Covariate search** ranges only over what was recorded — structured `obs_context`
components, not the hash. Hence §3.2, hence **log wide**: observations must capture
ambient state you currently have no reason to care about.

**Guard representation, decided (closes v0.1 open decision 1): binned comparisons plus
monotone bin constraints.** "T increases with P across these bins" is gateable with the
existing MDL machinery, captures the direction of most physical relations, and keeps
the engine inside stratified Datalog. Free-form `T = f(P)` means function symbols in
rule heads — an exit from Datalog into constraint-logic territory, named here as the
future engine change it is, and deferred.

**Refinement is specialization, not correction.** The original survives:

```
boils(water, 373.15K).                               -- general, retained
boils(water, T) :- pressure_bin(P), T in bin_f(P).   -- guarded refinement, higher
                                                     -- specificity, monotone in P
```

Conflict resolution by specificity ordering; the guarded rule takes precedence when its
body fires. Nothing is demoted. The fact was not wrong — it was **contextualized**.

**Gate guards hard, because this is a fishing expedition by construction.** With enough
candidate covariates, something always splits the data. Required:

- minimum support per partition (default ≥ 8 observations per side)
- held-out observations, not the discovery set
- **MDL**: `DL(guard) + DL(residual | guard) < DL(residual)` — the guard must buy more
  compression than it costs in description length
- multiple-comparisons correction (Benjamini–Hochberg) over the covariate set tested

**When nothing explains it**, the open question persists with its residual partition and
`ruled_out` set. This is the honest form of curiosity: an anomaly with a shape, not a
shrug. The residual cluster is also an **experimental design** — whatever the disagreeing
observations do share narrows the search, and the shape tells the agent what to go
measure. `suggested_measurement` is an actionable instruction, not a passive wait.

**Retroactive re-testing.** Because replay is from the log, when a new predicate appears
later (someone finally starts recording elevation), every open question is re-tested
against it automatically. Explanations arrive months after the confusion.

**Sideways suspicion.** When a latent is confirmed on one fact, co-derived facts and
siblings under the same rules are flagged as candidates for the same unrecorded argument.
Cheap to compute from the existing closure.

**Labeling.** A discovered guard is **conditioning, not causal**. Elevation and latitude
correlate in your sample. Stored and reported as an index, never as a mechanism;
prediction under intervention is explicitly unlicensed.

### 4.6 Support breadth — the exposure nothing else catches

Breadth of support is a **separate axis from count of support**, and it is where
dispersion detection is structurally blind.

A hundred confirmations all collected at sea level are indistinguishable from a hundred
collected across five elevations: same counts, same tight variance, same apparent
crystallinity. But the first fact is wrong-in-waiting and **no anomaly will ever fire,
because the confounder never varied.**

```
obs_context       = structured key/value ambient state per observation (§2)
context_sig       = hash(canonical serialization)      -- fast grouping only
breadth_key(k)    = normalized entropy of values of key k among confirming obs
breadth(fact)     = mean over keys with coverage ≥ floor;
                    distinct-signature count retained as a lower bound
breadth_class     = narrow | moderate | broad
transferability   = min(count_confidence, breadth_cap[breadth_class])
```

Per-key entropy is the change from v0.1: hash-distinct counting understated diversity
(two contexts differing in one irrelevant key look maximally different) and made
`context_sig` granularity a single fatal global knob. Computed per covariate, breadth
degrades gracefully instead — which mostly dissolves v0.1 open decision 2.

Low diversity caps transferability independently of count. `narrow` reads as *locally
reliable, extrapolation unlicensed*. This is the closest reachable bound on an unknown
unknown: you cannot know what you failed to record, but you can know that everything you
did record looked the same — and refuse to sell confidence outside that slice.

Low-breadth, high-dependent facts enter the eval queue as diversification targets:
*go confirm this somewhere else.*

### 4.7 Caveat propagation

Detection precedes identification, and that intermediate state is first-class knowledge:
*this fact varies for reasons I can measure but not name.* It has a magnitude, a residual
partition, and a ruled-out set. It attaches to the fact, not merely to a queue.

It then propagates. Any derivation touching an under-specified fact inherits the flag,
giving a **third derivation quality**:

| Quality | Condition | Contract |
|---|---|---|
| **proof** | exact edges, no flags | follows, full stop |
| **proof-modulo-unknown-context** | exact edges, ≥1 flagged or narrow-breadth premise | structurally valid, contingent on unenumerated conditions |
| **conjecture** | ≥1 soft edge | analogy with a similarity budget |

The caveat set additionally carries `shared_provenance` (§3.9) when premises trace to a
common source — correlated evidence dressed as independent confirmation is flagged at
the point of consumption.

Agents must receive the middle case distinctly, or they will act on a clean-looking proof
resting on a fact that is only reliably true where it happened to be tested.

---

## 5. API surface (agent-facing / MCP)

Distinct return types where the epistemic contract differs. This is load-bearing, not
ceremony.

```
recall(query, budget)          → prose entries + spans      [side-stream logged; no weight effect]
derive(goal, budget)           → Proof | NotEntailed | BudgetExceeded
conjecture(goal, sim_budget)   → Conjecture[]               [soft; never typed as Proof]
predict(stmt, budget)          → {p, ci, channels, sensitivity, mpe, caveats,
                                  snapshot_id, rejection_rate}
assert(stmt, source)           → candidate_id               [never a fact]
observe(stmt, outcome, ctx)    → event_id                   [ctx captured wide, structured]
observe_batch([...])           → event_id[]
claim(stmt, frame, criterion, due) → claim_id | Refused     [refused if unsettleable]
supersede(target, reason)      → event_id                   [sets valid_to on target]
pin(target, polarity, reason, authority) → event_id
why(id)                        → derivation + span provenance + gate_run + raw counts
                                 + composed counts + source diversity
questions(scope)               → open anomalies + pin tensions + suggested measurements
events_since(cursor, filter)   → event stream               [the ledger IS the outbox;
                                                             agents long-poll, no new
                                                             push machinery]
health()                       → calibration by partition, external ratio, invariant
                                 status + fail-policy state, breadth distribution,
                                 queue depth, quota exhaustion, constraint rejection rate
```

`derive`'s three-valued return is mandatory: `NotEntailed` may be returned **only** when
the stratified search exhausted within budget. Conflating "provably not derivable" with
"ran out of budget, unknown" is the sin the rest of the system exists to prevent, and
the engine always knows which case it is in. `predict` returning a `snapshot_id` is not
optional — it is the mechanism of I8.

---

## 6. Test harness

The harness is part of the spec, not an afterthought: a build is **conformant** when the
suite below is green against it. Reference skeleton: `candor_conformance.py`, which
defines the abstract `Driver` protocol all tests run against — any implementation
satisfying the protocol can be dropped under the same suite.

### 6.1 Conformance driver

The suite never touches implementation internals. It drives the system exclusively
through the §5 API plus four test-only hooks:

```
reset()                      -- fresh empty store
replay()                     -- force rebuild from ledger; returns closure hash
ledger_head()                -- current chain head hash
corrupt(what)                -- test-only fault injection: torn_tail | drop_index |
                                delete_payload(hash)
```

Hooks live behind a `HarnessDriver` extension of the protocol so production surfaces never
carry fault injection.

### 6.2 Property invariants (continuous, auto-derived)

Run continuously in CI and on a schedule in production. Property-based (Hypothesis)
where marked ⚡; golden fixtures otherwise. Each maps to an invariant and a fail policy.

| Family | Property | Inv | Policy |
|---|---|---|---|
| monotonicity ⚡ | adding support must not lower p | — | alert |
| irrelevance ⚡ | a fact outside the proof's support must not move the result | — | alert |
| permutation ⚡ | independent fact order cannot change a conclusion | — | alert |
| premise entailment | every emitted conclusion follows from its cited premises (kernel check) | — | **fail-stop** |
| pin integrity | no admitted structure violates a pin | I5 | **fail-stop** |
| count provenance | every `fact_counts` increment traces to an `observation`/`resolution` event id; retrieval stream has no write path (import-graph check + provenance scan) | I2 | **fail-stop** |
| count integrality | every stored count is an integer keyed by (actor, channel); no real-valued count exists in storage | I11 | **fail-stop** |
| composition purity ⚡ | cached `epi_*`/`alea_*` == recompute from `fact_counts` × reliability, always | I11 | alert |
| replay determinism | rebuild from ledger reproduces current closure bit-for-bit (post-redaction state) | I1, I3 | **fail-stop** |
| snapshot completeness | re-running `predict` at a claim's recorded ledger position with its pinned calib map reproduces `predicted_p` exactly | I8 | **fail-stop** |
| alias reversibility ⚡ | alias → query → supersede-alias → query restores pre-alias results bit-for-bit; counts never merged in storage | I11 | alert |
| valid-time isolation | superseded-regime facts never enter current-regime predictions; historical queries return regime-correct values | — | alert |
| budget honesty ⚡ | `derive` never returns `NotEntailed` when the search was truncated (engine exhaustion flag instrumented) | I4 | **fail-stop** |
| constraint conditioning | injected mutex-violating pair → violating worlds carry zero mass; `rejection_rate` reported and nonzero | — | alert |
| lexical firewall | `grep -rn weight` outside the committed tier returns nothing | I2 | alert |
| redaction integrity | chain verifies after payload deletion; replay excludes redacted content | I1 | **fail-stop** |

Fail-stop halts writers (gate + observation writers); alert-only pages. Policy state is
visible in `health()`.

### 6.3 Golden fixtures (unit)

- **Canonicalization:** `boils(water, 212F)` and `boiling_point(H2O, 100C)` + admitted
  alias → one canonical fact readable under either symbol; counts unioned at read only.
- **Gate steps:** one fixture per step 1–7 where exactly that step fails; admission
  requires all seven; rejection recorded with the failing step.
- **Demotion hysteresis:** alternating-outcome stream must not flap `structural`
  status; demotion fires only past the configured odds factor.
- **MDL guard:** paired fixtures — a guard that genuinely compresses (accept) vs an
  overfit split on the same support (reject); BH fixture with k spurious covariates
  shows bounded false discovery.
- **Changepoint vs guard routing:** a step-function outcome series routes to
  supersede-with-valid-time; a pressure-conditioned series routes to a guard candidate.
  Both repairs pass through the gate.
- **Credit assignment:** argmax-only integer blame; `definitional` / `exact`-derived /
  `frozen` targets never blamed; full sensitivity vector present in resolution payload.
- **Pin tension:** contradicting observations against a `-` pin are counted, the pin
  holds, a `pin_tension` question opens past threshold.
- **Two-channel routing:** an observation on a `crisp` fact moves epi only; on a
  `frequency` fact moves alea only; a validity resolution moves a frequency fact's epi.

### 6.4 Statistical validation (synthetic worlds, seeded)

A generator produces worlds with **known ground truth**; the system's estimates are
scored against it. All tests seeded and deterministic.

- **Two-channel recovery:** crisp facts with known truth values and frequency facts
  with known θ; posterior coverage of nominal intervals within tolerance (e.g., 90% CI
  covers ≥ 87% over 1k trials).
- **Dispersion power:** inject a latent binary covariate with effect size δ; measure
  detection power and false-positive rate at the Tarone threshold across n; power curve
  recorded as a regression artifact.
- **Changepoint power:** step function in θ at known t; detection latency and
  false-alarm rate under CUSUM defaults.
- **Breadth discrimination:** equal-count narrow vs broad confirmation sets must land
  in different `breadth_class` and different transferability caps.
- **Calibration loop:** claims generated with known p; reliability slope ≈ 1 per
  partition; isotonic map improves held-out Brier, never trains on read-path data.
- **Actor discount:** one honest actor + one random-outcome actor observing the same
  facts; after settlements score reliability, the composed posterior must sit closer to
  the honest-only posterior than to the naive pool.

### 6.5 Adversarial suite

- **Observation flooding:** within quota absorbed; beyond quota refused + event emitted.
- **Gate flooding:** candidate spam hits `cand_quota_per_epoch`; queue depth bounded.
- **Poisoned evidence:** a crafted evidence entry induces the extractor to propose a
  contradiction-bearing rule; held-out check or pin veto rejects it; rejection recorded
  with provenance to the poisoned span.
- **Hallucinating observer:** covered by 6.4 actor discount — damage bounded and
  reversible via recompute-with-exclusion.
- **Reputation farming probe:** actor builds `rel` on trivially-settled statements, then
  lies elsewhere. Marked `xfail-by-design`: the test documents the v1 limitation and
  pins its blast radius (quota × discount bound) so regressions in the mitigation are
  caught even though the attack is not fully defended.

### 6.6 Durability & replay

- **Torn tail:** kill mid-append; recovery truncates to last verifying line; no silent
  corruption; loss bounded by fsync policy.
- **Index loss:** delete SQLite entirely; rebuild from segments; closure hash identical.
- **Redaction replay:** delete a payload via redaction; chain verifies; replay excludes
  the content; determinism defined against post-redaction state.
- **Checkpoint honesty:** rebuild-from-checkpoint spot-verified against full replay;
  checkpoint carries the ledger head hash it summarizes.
- **Oracle reproducibility:** `deterministic_*` oracles re-run in pinned env
  (`env_hash`); non-reproduction demotes to `stochastic` automatically.

### 6.7 Stage gates

Each build stage (§8) exits when its tagged subset is green:

| Stage | Required green tags |
|---|---|
| 1 | replay determinism · count provenance · count integrality · redaction integrity · torn tail · index loss · calibration loop (trivial weights) |
| 2 | premise entailment · gate steps · canonicalization · lexical firewall · snapshot completeness |
| 3 | budget honesty · alias reversibility · composition purity · two-channel routing · actor discount · pin integrity/tension |
| 4 | monotonicity · irrelevance · permutation · constraint conditioning · two-channel recovery · calibration loop (full) · **the honest test (6.8)** |
| 5 | dispersion power · changepoint power · breadth discrimination · MDL/BH fixtures · changepoint-vs-guard routing · valid-time isolation |

### 6.8 The honest test — comparators defined now, not later

Pre-registered before Stage 4 begins, so the goalposts cannot drift under sunk cost:

- **Retrieval:** held-out QA set; nDCG@k and recall@k vs a plain embedding store over
  the same corpus.
- **Calibration:** Brier and log loss per partition vs the fair baseline — an embedding
  store cannot emit probabilities, so the comparator is **RAG with elicited
  probabilities plus isotonic fitting** on the same claims.
- **Margin:** the separation threshold that counts as "worth it" is written down before
  the run.

If Stage-4 calibration and retrieval quality do not separate from the baselines, the
architecture is elegant and unnecessary. Run the comparison before building Stage 5.

---

## 7. Scalability envelope

Rule count dominates, not fact count.

| Component | Behavior | Ceiling / mitigation |
|---|---|---|
| Closure materialization | fine to low millions of facts, single box | recursive rules with high fan-out are what explode; stratify, bound recursion depth, semi-naive eval |
| Behavioral signatures | sparse over rules, thousands of dims | scales well; LSH if rule count explodes |
| Calibration | bucket counts | free |
| **WMC × epistemic sampling** | **S × #P-hard — the real ceiling** | budget degrades S first, then exact → d-DNNF → top-k bounds; degrade, never hang |
| Actor-keyed counts | storage × active actors per fact | sparse in practice; actors-per-fact is small; composition cached |
| Replay from log | linear in *chained* events; retrieval volume no longer in the chain | checkpoint + incremental deletion (DRed) when replay exceeds SLA |
| Throughput bottleneck | **LLM extraction + sandboxed gate runs** | wall-clock bound, not compute bound; batch, cache, parallelize gate runs |

**Multi-agent:** admission is a serialization point. Shared committed tier, per-agent
evidence stores, **single writer to the gate, single sequencer for the ledger.**
Concurrent admission of contradictory rules leaves the closure briefly and provably
wrong. Calibration pools within `predictor_class`, never across (I9).

---

## 8. Build order

Ordered so that each stage makes the next one *verifiable*. Building fuzzy inference
first produces a system emitting confident numbers with no way to know whether they mean
anything — the exact failure this design exists to prevent. Stage exit = the §6.7 tag
set green.

**Seed path.** Initial rules, priors, the predicate registry bootstrap, and integrity
constraints enter via **human-as-proposer through the same gate**, never around it.
Stage 1 needs this path for its trivial-weight runs anyway; it costs nothing to say and
closes the "where does the first rule come from" hole.

**Stage 1 — Ledger, claims, calibration harness.**
Segments + CAS + index, redaction, torn-tail recovery, claim registry, calibration
bucketer on trivial weights (even all 0/1).
*You learn whether the system is honest before building machinery for it to be dishonest with.*

**Stage 2 — Promotion pipeline.**
Extractor → candidate → gate (all seven steps incl. canonicalization) → fact, with span
provenance and recompute-on-supersede. Proof checker kernel isolated and independently
tested. Predicate registry live; seed path exercised.

**Stage 3 — Identity, actors, soft edges.**
Actor-keyed counts + reliability scoring + read-time composition; aliases with
union-at-read; behavioral KNN as conjecture generator; ε-defaults via Dirichlet; hard
zero only from pins; three-valued `derive`.

**Stage 4 — WMC prediction, two-loop sampling.**
Exact enumeration + inclusion–exclusion first; compilation only above budget; constraint
conditioning with rejection accounting; isotonic recalibration at read time with map
hash in snapshots. **Run 6.8 before proceeding.**

**Stage 5 — Curiosity engine.**
Dispersion tests, changepoint detection with valid-time supersede, guard synthesis
(bins + monotone) with MDL gate, per-key breadth accounting, caveat propagation,
retroactive re-testing. Exit additionally requires: at least one guard discovered,
gated, and validated on held-out data; at least one open question explained
retroactively by a later-arriving covariate.

---

## 9. Decisions

### Closed in v0.2

1. **Guard representation** → binned comparisons + monotone bin constraints. Free-form
   `T = f(P)` requires function symbols in heads — an exit from Datalog, named and
   deferred. (§4.5)
2. **`context_sig` granularity** → dissolved by structured `obs_context` and per-key
   entropy breadth; no single global knob remains. Remaining tuning is the per-key
   coverage floor. (§4.6)
3. **Epistemic/aleatoric composition** → two-loop sampling: parameters and crisp truths
   sampled once per epistemic world, WMC within. Spread across worlds = epistemic;
   within-world mass = aleatoric. (§3.9)
4. **Evidence-tier `@weight`** → renamed `@salience`; ranking-only; firewall is lexical
   and audited. (§3.2, §6.2)
5. **Definitional boundary** → definitional iff derivable by the trusted core alone with
   no reference to counts and no world input. Closure membership fails the test and
   doesn't need to pass it — exact-derived facts are already blame-ineligible. The
   ambiguity was a false dilemma. (§2 schema comment, §4.4)
6. **Fuzzy fallback** → refused outright. `recall` is the sanctioned uncalibrated read
   path; there will not be a second one that returns number-shaped non-probabilities.
   (§0)

### Open (new)

1. **Reputation-farming defense** beyond quotas + frame-partitioned reliability —
   per-predicate reliability? stake-weighted scoring? Deferred; blast radius pinned by
   test 6.5.
2. **Scalar/continuous outcomes** — predictive densities + CRPS scoring. Deferred; v1
   is binary by scope (§0).
3. **Changepoint defaults** — CUSUM threshold and BOCPD hazard prior need empirical
   tuning; ship CUSUM-first, revisit with data.
4. **Per-key breadth coverage floor** and aggregation weighting — empirical; the
   regression artifact from 6.4 is the tuning instrument.
5. **d-DNNF compiler** — integrate an existing compiler (c2d/dsharp) vs in-house top-k
   only. Decide when the Stage-4 budget numbers exist.
6. **Alias similarity threshold** for behavioral-basis admission — tune against 6.3
   canonicalization fixtures once real signatures exist.

---

## Appendix A — v0.1 → v0.2 changelog

| Area | Change | Motivated by |
|---|---|---|
| Trust model | Actor-keyed integer counts (`fact_counts`), reliability scored only via trusted settlements, read-time discount, quotas; new I11 | Review §1 (observation side door) |
| Channels | `stmt_type` axis; explicit event→column mapping; two-loop sampling composition | Review §2; closes old open decision 3 |
| Identity | Predicate registry (typed, unit-canonical args), gate canonicalization step, `alias` events with union-at-read, cold-start statement | Review §3 |
| Time | Time always a covariate; changepoint machinery; supersede-with-valid-time; `valid_from/valid_to`; no-decay policy stated | Review §4 |
| Context | Structured `obs_context`; `context_sig` derived; per-key entropy breadth | Review §5; dissolves old open decision 2 |
| API | Three-valued `derive`; budget params; `supersede`/`pin`/`observe_batch`/`events_since` verbs; snapshot contents defined; verifier/env hashes on resolution | Review §6 |
| Contradiction | First-class `constraints` (mutex, functional); gate step 7 defined; constraint-conditioned WMC with rejection rate; pin-tension questions | Review §7 |
| Statistics | `shared_provenance` caveat; calibration partition gains `predictor_class` + min-n; argmax integer blame with logged sensitivity vector; binary-outcome scoping | Review §8 |
| Plumbing | Ledger = JSONL segments + CAS + derived index; payload commitments + redaction; retrieval side stream; span anchors `(entry_id, content_hash, offset)`; invariant fail policies; sandbox hermeticity + oracle demotion | Review §9 |
| Decisions | Old §9 closed (6/6); new open list; honest-test comparators pre-registered; seed path through gate | Review §10 |
| Testing | §6 replaced by full conformance harness: driver protocol, 16 property invariants, golden fixtures, seeded statistical validation, adversarial suite, durability suite, stage gates | This revision |

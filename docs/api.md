# API guide

Everything goes through `candor.system.CandorSystem`. Return types differ
where the epistemic contract differs — that's load-bearing, not ceremony.

```python
from candor.system import CandorSystem
m = CandorSystem("./store")     # creates or reopens; replays the ledger
```

## Writing

### `assert_(stmt, source, actor) -> candidate_id`
Proposes. Never creates a fact directly (the LLM lives at the candidate
boundary, and so do you). Statement forms:

```python
m.assert_({"pred": "boils_at", "args": ["water", "212F"], "stmt_type": "crisp"},
          source="doc:phys-101", actor="agent:reader")
# unknown predicates auto-propose a symbol candidate alongside the fact
m.assert_({"kind": "symbol", "pred": "boils_at", "arity": 2,
           "arg_types": ["substance", "temperature"],
           "canonical_units": {"1": "K"}}, source="seed", actor="human:me")
m.assert_({"kind": "constraint", "ctype": "mutex",
           "body": {"pred": "link_state", "exclusive_values": ["up", "down"]}},
          source="seed", actor="human:me")
m.assert_({"kind": "rule", "head": {"pred": "reachable", "args": ["?x", "?z"]},
           "body": {"literals": [{"pred": "reachable", "args": ["?x", "?y"]},
                                 {"pred": "reachable", "args": ["?y", "?z"]}]},
           "holdout": {"hits": 5, "misses": 0}}, source="seed", actor="human:me")
m.assert_({"kind": "alias", "canonical": "boils_at", "alias": "boiling_point",
           "basis": "pinned"}, source="review", actor="human:me")
```

### `run_gate() -> list[run]`
Runs the curiosity sweep, then drains all pending candidates through the
seven admission steps. Each run reports `candidate_kind`, `status`
(`admitted`/`rejected`), and the `failing_step` + reason on rejection.
Units are canonicalized at admission: `212F` is stored as `373.15K` and the
fact is readable under either notation. The sweep is batch-triggered, so a
fact's breadth/dispersion flags — and therefore `closure_hash()` — only reflect
the latest observations after a `run_gate()` or a reopen; call it before
comparing closure hashes across processes (see
[architecture.md](architecture.md#the-curiosity-engine-stage-5)).

It also opens with the **prospective audit** (v0.6 Δ11): every admitted guard is
scored on the observations that arrived *after* its admission, and one whose
direction reverses (at the §3.4 hysteresis bar) or stops beating chance (on
twice the entry evidence) is **demoted** through the ledger — rule out of the
closure, candidate row closed, read paths silent. Demotions appear in the
returned runs as `status: "demoted"`. Re-entry is judged on post-demotion
evidence only, so a demoted guard returns on new data or not at all.

### `observe(stmt, outcome=None, ctx, actor, value=None, confidence=None) -> event_seq`
An attributed outcome report. `ctx` is free-form key/value ambient state —
**log wide**; it's the raw material for finding missing variables later.
`confidence` (0..1) grades the vote; the API bins it to an integer grade and
keeps the raw value in the event payload. For a **categorical** fact, pass
`value="captcha"` (the realised category) instead of `outcome` — which field is
authoritative is decided at fold time by the fact's `stmt_type`. `observe_batch([...])`
for bulk.

Two reserved `ctx` conventions (v0.6): keys prefixed **`do:`** record that the
agent was *acting on* the world, not merely watching it (Δ13) — a guard found
on one is labeled regime dependence, and predictions pooling across mixed `do:`
regimes carry a `regime_mixed` caveat. Keys prefixed `derived:` are
system-synthesized (Δ10) and never yours to write. Even with an empty `ctx` the
sweep still searches the synthesized frames — hour-of-day and day-of-week from
the event timestamp, the fact's own previous outcome, and pairwise interactions
of whatever keys you did record.

### `claim(stmt, frame, criterion, due) -> claim_id | "Refused"`
Registers a prediction to be settled. `frame` is `internal`/`external`;
`criterion` names a registered oracle (`register_oracle(...)` first for
external verifiers). A claim with no constructible verifier is **refused** —
unsettleable statements stay prose.

### `resolve(claim_id, outcome=None, value=None, ...) -> event_seq`
Settles a claim. This is the **only** path that moves trust: every prior
observation on the statement is scored against the settled outcome, updating
each actor's confusion and response ledgers. For a categorical claim, pass the
realised `value`; its surprisal is `-log P_frozen(value)`, which stays **finite
even for a value never seen before** (it scores against the frozen unknown mass)
— trust moves per source via a one-vs-rest reduction.

### `pin(target_id, polarity, reason, authority)` / `supersede(target_id, reason)`
A `-` pin is the system's only hard zero (contradicting observations are
counted, the pin holds, and past a threshold a human gets paged via
`questions()`). Supersede reverses anything — facts, aliases, pins — by
appending, never by editing.

### `retract_source(actor, reason, restore=False) -> event_seq`
Silences one source. Its event skeletons stay in the chain; they stop
contributing, and every downstream number — counts, trust, predictions —
recomputes as if it never spoke. Append-only and reversible. **This is how you
recover from a bad source.**

### `redact(payload_hash, authority="human:operator") -> event_seq`
Deletes a payload; the chain still verifies; replay recomputes all state
without the content. Scoped to *content*, not to a source: payloads are
content-addressed and carry no actor, so every event sharing the hash loses its
payload whoever wrote it. Use it for secrets and PII, check
`redaction_scope(payload_hash)` first, and use `retract_source` for a bad actor.

### `set_authz(policy)` — opt-in access control
By default `authority`/`actor` are attribution labels, not authenticated
identities (the trust boundary is the process; see [SECURITY.md](../SECURITY.md)).
Register a policy `(principal, op) -> bool` and CANDOR enforces it on the
privileged writes — `pin`, `redact`, `retract_source`, `register_oracle`,
`set_reliability` — raising `Unauthorized` **before** any ledger append, so a
denied call mutates nothing. `set_authz(None)` restores advisory mode. The
policy is runtime config, never part of replay: an existing ledger is never
re-checked.

## Reading

### `predict(stmt, budget) -> PredictOutcome`
`p`, `ci`, `channels` (epistemic vs aleatoric spread), `sensitivity` (which
fact flips the conclusion), `mpe`, `caveats` (e.g. `shared_provenance`,
`narrow_breadth`, `unstable` for a flaky fact, `regime_mixed` when the number
pools observations across a `do:` intervention boundary — Δ13), `rejection_rate`
(constraint tension), and `snapshot_id`. `predict_at(stmt, snapshot_id)` re-runs
at a recorded ledger position and reproduces the number exactly.

### `predict` on a categorical fact -> `CategoricalPrediction`
When `stmt` names a categorical fact, `predict` returns a **distribution**, not a
scalar: `values` (`{value: slice}` in canonical order, each slice a point `p`
with a `ci`), `unknown` (the never-seen mass as its own slice), `total_observations`,
`snapshot_id`, `caveats`, and `by_context` (the value distribution conditioned on
a discovered guard key — each context group carrying its own unknown slice plus a
`__residual__` group for observations that didn't record the key). `Σ values[*].p
+ unknown.p == 1.0` exactly. The unknown mass is a Dirichlet-process / CRP
predictive `alpha/(N+alpha)`: thin data reserves more unknown, and it shrinks as
observations accumulate. `predict_at` reproduces the full distribution
bit-for-bit and `alpha` rides the snapshot id (see the
[v0.5 delta](spec-v0.5-delta.md)).

### `distribution(stmt) -> dict`
A pure read-time breakdown of a flaky **binary** (crisp/frequency) fact — it
writes nothing, moves no count, changes no `closure_hash`, and never runs
`predict`. `modes` splits the true-rate by each recorded context key (with a
`__residual__` bucket for observations that didn't record it); `derived_modes`
(Δ10) does the same for the synthesized frames (`derived:hour`, `derived:prev`,
…); `residual` reports how much of the spread an admitted guard **explains**
(`explained` = the η² correlation ratio of its partition, computed on the
augmented projection when the guard's key is derived) versus the honest
`unexplained` remainder. It is the companion to the `unstable` caveat: not just
*that* a fact is flaky, but *how* it splits by context and how much no recorded
variable accounts for.

### `derive(goal, budget) -> DeriveOutcome`
Three-valued, honestly: `proof` (with a kernel-checked derivation and a
quality tag — `proof`, `proof-modulo-unknown-context`, or `conjecture`),
`not_entailed` (**only** when the search space was exhausted), or
`budget_exceeded` (truncated ⇒ unknown — never conflated with absence).

### `conjecture(goal, sim_budget, commit=False) -> [conjectures]`
Analogical proposals over behavioural signatures (§4.3): for a goal `P(args)`,
a predicate that *fires like* `P` and already holds at those args is offered as
`P(args)` by analogy — never typed as a proof, blind below the signature floor,
refused above the similarity budget. With **`commit=True`** (v0.6 Δ12) each
proposal is filed as a **claim** by `agent:conjecture`: `predicted_p` is the
analog's own earned probability (the transfer moves an audited number —
similarity is not a probability), under the distinct `conjecture/v1` predictor
class so the analogy engine's calibration curve is measured on its own (I9).
`resolve(claim_id, outcome=True)` then asserts the goal as an ordinary fact
candidate through the gate — postulate, validate, implement, in that order and
never out of it.

### `recall(query, budget) -> [entries]`
Prose retrieval over the evidence tier: BM25 + RM3 expansion + sub-token
indexing, optionally fused with a dense ranker (see below). Side-stream
logged; cannot move any number.

### `why(fact_id) -> dict`
The full audit: raw per-actor counts, composed view, gate run, span
provenance, context-key diversity, current derivation and its quality.

### `questions(scope)` / `health()` / `events_since(cursor, kinds)`
Open anomalies with suggested measurements; calibration by partition, quota
and breadth status; and the ledger as an outbox for long-polling consumers.

## Configuration

```python
m.set_actor_quota("agent:bulk", obs_per_epoch=100_000, cand_per_epoch=10_000)
m.register_oracle("verifier:ci", "deterministic_total", impl_ref, code_hash, env_hash)
```

**Optional dense retrieval** (embeddings never enter the trusted core; the
embedder is injected, and its absence degrades to lexical):

```sh
CANDOR_EMBED_URL=http://your-ollama:11434 CANDOR_EMBED_MODEL=bge-m3:latest python app.py
```

## Test-only surface

`CandorHarnessDriver` (see `src/candor/harness.py`) adds `reset`, `replay`,
`corrupt` (torn tail / index drop / payload deletion) and introspection hooks.
It exists for the conformance suite; production surfaces never carry fault
injection.

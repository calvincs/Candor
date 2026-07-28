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
fact is readable under either notation.

### `observe(stmt, outcome, ctx, actor, confidence=None) -> event_seq`
An attributed outcome report. `ctx` is free-form key/value ambient state —
**log wide**; it's the raw material for finding missing variables later.
`confidence` (0..1) grades the vote; the API bins it to an integer grade and
keeps the raw value in the event payload. `observe_batch([...])` for bulk.

### `claim(stmt, frame, criterion, due) -> claim_id | "Refused"`
Registers a prediction to be settled. `frame` is `internal`/`external`;
`criterion` names a registered oracle (`register_oracle(...)` first for
external verifiers). A claim with no constructible verifier is **refused** —
unsettleable statements stay prose.

### `resolve(claim_id, outcome, ...) -> event_seq`
Settles a claim. This is the **only** path that moves trust: every prior
observation on the statement is scored against the settled outcome, updating
each actor's confusion and response ledgers.

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

### `redact(payload_hash) -> event_seq`
Deletes a payload; the chain still verifies; replay recomputes all state
without the content. Scoped to *content*, not to a source: payloads are
content-addressed and carry no actor, so every event sharing the hash loses its
payload whoever wrote it. Use it for secrets and PII, check
`redaction_scope(payload_hash)` first, and use `retract_source` for a bad actor.

## Reading

### `predict(stmt, budget) -> PredictOutcome`
`p`, `ci`, `channels` (epistemic vs aleatoric spread), `sensitivity` (which
fact flips the conclusion), `mpe`, `caveats` (e.g. `shared_provenance`,
`narrow_breadth`), `rejection_rate` (constraint tension), and `snapshot_id`.
`predict_at(stmt, snapshot_id)` re-runs at a recorded ledger position and
reproduces the number exactly.

### `derive(goal, budget) -> DeriveOutcome`
Three-valued, honestly: `proof` (with a kernel-checked derivation and a
quality tag — `proof`, `proof-modulo-unknown-context`, or `conjecture`),
`not_entailed` (**only** when the search space was exhausted), or
`budget_exceeded` (truncated ⇒ unknown — never conflated with absence).

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

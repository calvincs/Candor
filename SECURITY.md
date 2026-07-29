# Security model

CANDOR is a substrate an application embeds, not a multi-tenant service. This
document states precisely what it does and does not defend against, so the
embedding application can supply what it must.

## The trust boundary is the process

The `actor`, `source`, and `authority` strings on every write are **attribution
labels**, not authenticated identities. By default CANDOR does not verify that
the caller passing `authority="human:operator"` is an operator — it records the
claim and moves on. Anyone who can call the Python API can write any label.

This is deliberate: CANDOR's job is to keep an auditable, replayable epistemic
record, and it draws its trust boundary at the process. Deciding *who* may open
the store and *which* calls they may make is the embedding application's
responsibility — the same way SQLite trusts whoever holds the file handle.

What the substrate does enforce, regardless of labels:

- **Trusted core vs. untrusted periphery** (LCF-style, checked by `make audit`):
  nothing in `periphery/` can write a committed number; retrieval has an empty
  import list so no code path reaches the count updater.
- **The admission gate** validates every candidate — including synthesized
  verifiers, which run in a hardened sandbox (AST allow-list, no dunder/import,
  `setrlimit`, `-I -S`) before they can ever be admitted as an oracle.
- **Quotas** (§3.12) bound how much any one actor can write per epoch, so a
  single label cannot flood the store even in advisory mode.

## Opt-in enforcement of privileged writes

If your deployment *does* have a notion of authenticated principals, register a
policy and CANDOR will enforce it on the privileged writes — `pin`, `redact`,
`retract_source`, `register_oracle`, and `set_reliability`:

```python
OPERATORS = {"human:operator", "svc:governance"}
m.set_authz(lambda principal, op: principal in OPERATORS)

m.pin(fact_id, "+", "trusted", authority="mallory")          # -> Unauthorized
m.pin(fact_id, "+", "trusted", authority="human:operator")   # allowed
```

The policy is a callable `(principal, op) -> bool`. It is consulted at the API
boundary **before** any ledger append, so a denied call mutates nothing — the
ledger head does not move, and the attempt is recorded in `health()` for audit.
`set_authz(None)` restores the default advisory posture.

The policy is *runtime configuration, not ledger state*. It never enters the
closure hash, and **replay of an existing ledger never re-checks it**: every
event already in the chain was admitted under whatever policy (or none) was
active when it was written. Rotating or tightening the policy therefore changes
no historical number — it only governs new writes. Authenticate the principal
however your deployment does (mTLS, signed tokens, an OS uid check) and have the
policy consult that; CANDOR only asks the yes/no question.

## The hash chain is consistency, not tamper-evidence

Each event commits to its predecessor's hash. `verify_chain()` therefore
detects the failure modes that **lose or scramble** data — accidental
corruption, torn writes, and two processes forking the same log. It does **not**
prove the log was never rewritten: an adversary who can edit a ledger segment
can recompute every hash after their edit, and the chain will still verify.

If your threat model includes a writer who can rewrite the ledger, anchor it
externally — periodically sign or notarise the current head (`ledger_head()`)
somewhere the writer cannot reach, and check it on open. CANDOR gives you the
head to anchor; it does not manage keys or an external log for you.

## Redaction and retraction

- `redact(payload_hash)` purges a *content-addressed payload* everywhere it
  appears. Because identical reports share one payload, redacting a hash also
  destroys honest reports that happened to agree — call `redaction_scope` first.
- `retract_source(actor)` silences a *source*; its blast radius is exactly that
  actor, and it is reversible (`restore=True`). This — not `redact` — is how you
  recover from a bad source.

Both leave the event skeletons in the chain (the audit trail survives) and
recompute all downstream state without the excluded input.

## Reporting

This is a research substrate under active development. If you find a security
issue, open an issue or contact the maintainer rather than filing a public PR
with a working exploit.

# CANDOR — spec v0.4 delta

Amends v0.3. Motivated by 6.8-v2 (FINDINGS F7/F8): the mechanism wins wherever
≥2 semantic witnesses exist; the loss is information starvation — binary votes
discard most of what a judge knows, and random panels leave facts unwitnessed.

## Δ6 — graded observations (categorical Dawid–Skene)

`observe()` gains optional `confidence ∈ [0,1]`. The API boundary bins it into
an integer grade (0 = ungraded/legacy, 1 = weak, 2 = firm, 3 = strong, by
max(c, 1−c) < .75 / < .9 / ≥ .9); direction stays the boolean vote. Raw
confidence lives in the event payload (audit); the derived index stores only
the integer grade (I11).

New integer table, moved only by trusted settlements (§3.12):

    actor_response(actor, frame, vote, grade, n_true, n_false)

Read-time composition: a response (vote, grade) carries
LR = P(response|true)/P(response|false), Dirichlet-smoothed over the actor's
response ledger; actors with fewer than 10 scored responses fall back to the
Δ1 binary confusion LR, which itself falls back to the informative prior —
evidence can always speak (I5). Δ2 context grouping applies unchanged.
Graded LRs compose at posterior mean (documented simplification; binary votes
keep per-world parameter sampling).

## Δ7 — witness floor (test/deployment policy, §3.11 rationale)

Random panels left 15% of facts with ≤1 semantic witness. The eval scheduler
exists to route observation effort at under-evidenced targets; v3 panels
guarantee 2 semantic judges + 2 others, seeded and pre-registered. In
deployment this is a queue policy, not a mechanism.

## Noted for later rounds (principal's "learning" direction)

- **Observer-pool selection** (genetic flavor): actors whose learned response
  LRs converge to 1 are dead capacity; §3.11 scoring should reclaim their
  panel slots. Reported as a diagnostic this round, not yet a mechanism.
- **Per-predicate reliability** (spec §9 open) and **Stage-5 guard discovery**
  are the substrate-native forms of "learning that strengthens outcomes":
  conditions, not just weights. Unblocked by a 6.8 pass.

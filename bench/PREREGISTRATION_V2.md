# Test 6.8 v2 — pre-registration

> **Publication note (2026-07-27).** Internal LAN hostnames in this document
> were genericized for public release (`<gpu-host>` served ollama;
> `<crawl-host>` served the crawler). This is the ONLY edit ever made to this
> file after it was frozen, it changes no recorded number or margin, and it is
> called out here rather than applied silently. Because the verdict scripts
> hash-check the frozen originals, those originals remain byte-identical in
> the private research branch; hash checks against public copies will
> correctly report the file as modified.


STATUS: FROZEN — margins confirmed by the principal 2026-07-26, before any
v2 result existed. Not to be edited after this point; `run_6_8_v2.py` records
this file's SHA-256 and `verdict_v2.py` refuses to rule if it changes.

Supersedes nothing: the v1 result (`results_6_8.json`, commit 5b5cd57) stands
on the record as the outcome for the v0.2-mechanism build on suite v1. This is
a new experiment for the v0.3 mechanism (Δ1–Δ4) on suite v2, with margins
frozen before any result exists. The methodology is fixed: mechanisms and
tests may iterate between rounds; **margins never move after a run begins,
and a failed round is reported as failed.**

## What changed since v1, and why (evidence: bench/FINDINGS_6_8.md)

| | v1 | v2 | reason |
|---|---|---|---|
| mechanism | one-coin reliability (v0.2 §3.12) | two-coin confusion + context-grouped composition (v0.3 Δ1/Δ2) | one scalar cannot represent asymmetric observers; correlated votes double-counted |
| panel | fixed 4 observers, all vote on everything | 12 observers, random 4-of-12 per claim, assignment pre-registered in the suite artifact | fixed panels make vote patterns a sufficient statistic; isotonic-on-mean is then near-optimal by construction and attribution has no headroom |
| false claims | polarity flips 47/120 | per-kind cap ⅓; adds `within_entry_swap` whose substituted value IS present in the gold entry | v1's dominant kind defined one tool's confusion; within-entry swaps defeat presence checks and force reading |
| size | 240 claims (127 held out) | ~1,200 claims (~600 held out) | per-item Brier-diff SD 0.0876 → resolving 0.01 needs ~600 held-out |
| retrieval | plain BM25 | + sub-token indexing + RM3 (R1) | v1 lost on matching (gold absent 37 vs 14) |

## Systems

* **CANDOR** — the substrate at HEAD: two-coin confusion learned only from
  train-split settlements through the `deterministic_total` oracle,
  context-grouped sub-additive composition (γ = 0.5), per-world actor-parameter
  sampling, read-time isotonic fitted on train.
* **Baseline** — §6.8's comparator, unchanged: bge-m3 top-8 RAG + elicited
  probability (`qwen3.6:27b-q8_0`, the v1-vindicated judge: 97.5% agreement
  with laguna, marginally higher accuracy) + isotonic fitted on train.
* **Uniform control** — mandatory forever (Δ3): identical votes, discount off,
  unweighted mean + the same train-fitted isotonic. The mechanism must beat
  its own degenerate case or the failure is the result.

Leakage discipline as v1: held-out claims are never resolved anywhere; the
train/held-out split is `sha256("20260725" + claim_id) % 100 < 50`.

## Margins — CONFIRMED BY THE PRINCIPAL (2026-07-26)

The v1 bars (never passed, therefore not weakened) plus the Δ3 control bar,
selected explicitly and gating:

```
RETRIEVAL   candor.ndcg@10   >= baseline.ndcg@10   - 0.05
            candor.recall@10 >= baseline.recall@10 - 0.05

CALIBRATION brier(baseline) - brier(candor) >= 0.02
            paired-bootstrap 95% CI lower bound > 0
            candor ECE <= 0.10

NEW (Δ3)    brier(candor) <= brier(uniform_control) + 0.005
            — the trust machinery may not cost accuracy against its own
              knob-off case. GATING.
```

Retrieval posture: Δ5 decided as R2 (dense as untrusted periphery ranking
input; fusion of lexical+RM3 with bge-m3 under RRF; embeddings never in the
trusted core or committed tier; I2 intact). CANDOR's retrieval arm runs with
`CANDOR_EMBED_URL=http://<gpu-host>:11434`.

Suite v2 final, generated before this freeze: 1,194 items, 1,194 claims
(597 true / 597 false), kind caps honored (polarity_flip 199, numeric_swap
133, entity_swap 133, within_entry_swap 132), panels pre-assigned in
`suite_v2.json`.

6.8-v2 PASSES only if all bars hold. On failure: stop, report,
diagnose, and the next round changes mechanism or test — never these numbers.

## Reproduction

```
python -m bench.generate_suite_v2 1600     # suite_v2.json (seed 20260726)
python -m bench.run_6_8_v2                 # refuses unless STATUS: FROZEN
python -m bench.verdict_v2                 # applies the frozen margins
```

All model calls at temperature 0, seed 7, disk-cached; observer panel
assignments are inside `suite_v2.json`, derived from
`sha256(seed | "panel" | claim_id)`.

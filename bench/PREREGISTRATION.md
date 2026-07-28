# Test 6.8 — pre-registration

> **Publication note (2026-07-27).** Internal LAN hostnames in this document
> were genericized for public release (`<gpu-host>` served ollama;
> `<crawl-host>` served the crawler). This is the ONLY edit ever made to this
> file after it was frozen, it changes no recorded number or margin, and it is
> called out here rather than applied silently. Because the verdict scripts
> hash-check the frozen originals, those originals remain byte-identical in
> the private research branch; hash checks against public copies will
> correctly report the file as modified.


**Frozen before the run. Committed before any result existed.**

Spec §6.8 requires the comparators and the separation threshold to be written
down before Stage 4 concludes, "so the goalposts cannot drift under sunk cost".
The margins below were chosen by Calvin (the principal), not by the
implementation, and are not to be edited after results are seen. If the run
fails, the failure is the result.

Frozen: 2026-07-25.
Margins set by: Calvin (project principal), via explicit selection.
Implementation state at freeze: Stages 1–4 green, Stage 5 not built.

---

## 1. Corpus

| | |
|---|---|
| Source | `<local corpus A: agent memories>` (Pernix agent memories) + `<local corpus B: knowledge base>` (hyperkb) |
| Entries | 3,715 after de-duplication by content hash |
| Size | 1,877,271 characters |
| Split | pernix 3,424 · hkb 291 |
| Build | `python -m bench.corpus` → `data/bench/corpus.jsonl` |

Real prose written by a working agent over months, not text generated for this
test. Both systems index exactly the same entries.

## 2. Ground truth — how each label is provable

No label is taken on an LLM's word. The model proposes; a corpus scan disposes.
Anyone can re-check any item with `grep`.

**Retrieval gold.** `laguna-s-2.1:q8_0` reads one entry and emits a question, a
verbatim answer span, a self-contained fact, and an anchor token. An item is
kept only if all of the following hold mechanically:

1. the answer span occurs **verbatim** in its source entry (exact, normalized
   whitespace/case);
2. the span occurs in **≤ 3 entries** corpus-wide — the gold set is exactly the
   set the scan returns, not an opinion;
3. the anchor token occurs in the source entry and in ≤ 40 entries overall;
4. the question shares **no n-gram longer than 7 tokens** with its source entry.
   This is a symmetric difficulty filter — a question that copies a long run
   from the entry is trivial for lexical and dense retrieval alike — and it is
   applied before any system sees the item.

**Claim outcomes.** Half true, half false, alternating by position.

* TRUE — restates a span already proved present in the corpus.
* FALSE — a **programmatic, type-preserving** perturbation of the true fact: a
  quantity moves to a different plausible quantity (ranges stay well-ordered), or
  a same-class entity is transplanted from a different entry. Kept only if the
  substituted value appears **nowhere in the gold entries** and the mutated
  sentence matches **nothing corpus-wide**.

Type preservation matters: a malformed false claim ("873-115 tokens per second")
would test whether a system can spot garbled text, not whether it knows what the
corpus says.

## 3. The two systems

**CANDOR** — the build at the committed HEAD. Retrieval is `recall()` over the
evidence tier (BM25 + `@salience`, no embeddings anywhere). Calibration is the
architecture doing its actual job: four attributed observers vote, their votes
land as `observation` events keyed by `(fact, actor, channel)`, reliability is
learned **only** from training claims settled through a `deterministic_total`
oracle (§3.12), and the read-time isotonic map is fitted on the training split
only (§3.9, I8).

**Baseline** — the comparator §6.8 names. Dense retrieval with `bge-m3` over the
same corpus, top-k. For calibration, §6.8 is explicit that an embedding store
cannot emit probabilities, so the comparator is **RAG with elicited
probabilities plus isotonic fitting**: retrieve top-8, ask `laguna-s-2.1` for a
probability, fit isotonic on the training split, apply to held out.

**Naive-ensemble control** — not requested by §6.8, reported anyway because
omitting it would flatter CANDOR. Same four observers, unweighted mean, isotonic
from the same training split, **no reliability discount**. CANDOR beating the
baseline could be nothing but the ensemble; CANDOR beating *this* is what the
trust machinery actually buys.

**Leakage control.** Ground truth enters the store at exactly one point: the
`resolve()` call on training claims. Held-out claims are never resolved, never
settled, and never scored inside the system. Both isotonic maps are fitted on
the training split alone. The train/held-out split is `sha256("20260725" +
claim_id) % 100 < 50` — deterministic and independent of any result.

## 4. Margins — the goalposts

### Retrieval: non-inferiority at 0.05

```
PASS if   candor.ndcg@10   >=  baseline.ndcg@10   - 0.05
   and    candor.recall@10 >=  baseline.recall@10 - 0.05
```

Rationale, recorded now: §0 states CANDOR is "not a vector store, not a
replacement for one" and forbids embeddings in the core. Demanding that it beat
a dedicated dense retriever would contradict the spec's own non-goals. The
claim under test is that the provenance chain costs you little retrieval, not
that it wins retrieval.

### Calibration: superiority at 0.02, and a standalone honesty bar

```
PASS if   brier(baseline) - brier(candor)  >=  0.02
   and    paired-bootstrap 95% CI lower bound  >  0
   and    candor ECE  <=  0.10
```

The ECE bar is absolute on purpose: beating a badly calibrated baseline is not
the same as being calibrated, and §6.8 is a test of honesty rather than of
relative standing.

### Secondary (reported, not gating)

```
brier(control) - brier(candor)   what the reliability discount buys over
                                 plain ensembling
learned reliability per actor    agent:optimist must score visibly below
                                 tool:exact, or §3.12 is decoration
```

### Overall verdict

**6.8 PASSES only if the retrieval bar and the calibration bar both hold.**

## 5. On failure

Per §8 ("Run 6.8 before proceeding") and by explicit instruction: **stop.**
Write the results and a diagnosis, commit the failing numbers, do not begin
Stage 5, and hand the decision back. The margins are not renegotiated after the
fact — a pre-registered test that cannot stop the project is theatre.

## 6. Reproduction

```sh
python -m bench.corpus data/bench/corpus.jsonl   # 3,715 entries
python -m bench.generate_suite 320               # suite.json + rejects.json
python -m bench.run_6_8                          # results_6_8.json
```

Generation and judging run at `temperature=0` with a fixed seed on
`<gpu-host>:11434`, and every model call is cached on disk under
`data/bench/cache/`, so the run is re-executable without re-rolling the models.
`run_6_8.py` records the SHA-256 of this file in its output; if the hash in
`results_6_8.json` does not match this file, the goalposts moved and the result
is void.

---

## Amendment 1 — suite-authoring model (2026-07-25)

**Made before any result existed.** No arm of 6.8 had been run, no score of any
kind had been computed, and §4 (the margins) is untouched.

**Change.** Suite authoring moves from `laguna-s-2.1:q8_0` to
`qwen3.6:35b-a3b-q8_0`. `laguna-s-2.1` remains the baseline's probability
elicitor and the `agent:llm_big` observer — unchanged.

**Why.** `laguna-s-2.1` is a 117B dense model running at roughly 14 tok/s on
this box; a single suite-authoring call takes ~100s and the full 320-item
authoring pass was tracking at ~0.3 items/min, i.e. days. The same prompt on
the MoE model (3B active parameters) completes in 1.5s. Measured, not guessed.

**Why this does not touch the integrity of the test.** The authoring model's
quality is a *throughput* concern by construction, not a correctness one. Every
proposed item is verified against the corpus by exact string match (§2), so a
weaker author lowers the **yield of valid items** and cannot produce an invalid
label. The model that has to be strong is the one the baseline uses to elicit
probabilities, because a weak elicitor would make §6.8 a strawman — and that
model is unchanged.

**What would have been illegitimate**, and was not done: changing the model
used by either system under test, changing the margins, or making any change
after seeing a score.

---

## Amendment 2 — perturbation implementation (2026-07-25)

**Made before any result existed.** Still no arm run, no score computed, §4
untouched. This amendment changes no stated requirement; it fixes an
implementation that did not deliver what §2 already promised.

§2 says false claims are **type-preserving** perturbations. Inspection of the
generated suite showed three ways the code broke that promise, all fixed:

1. `5.19s` and `0.3` were classified as hostnames (the pattern accepted an
   all-digit label), so a hostname could be replaced by a duration —
   `rrstar.com → 5.19s`. A hostname now needs an alphabetic final label.
2. `news-summary-2026-05-29.md` was classified as a hostname for the same
   reason. Filenames are now their own class.
3. Transplants were permitted between `identifier` and `phrase` anchors, which
   lump proper nouns, file paths, regexes and product names together —
   producing visible word salad (`Rousseau → factcheck_manifest.json`).
   Transplants are now restricted to classes where members are genuinely
   interchangeable.

Two additions in the same spirit: a **polarity flip** (negating one unnegated
auxiliary) as a further type-safe falsification, and year-aware numeric
perturbation, because scaling `2025` by 2.5 gives `5062` — false, but not
plausible.

**Class balance.** §2 says "half true, half false". The implementation assigned
alternately *before* attempting perturbation, so every failed perturbation
silently dropped a false claim and skewed the base rate — the delivered suite
was 121 true / 50 false. The pool of falsifiable items is now computed first and
the split assigned afterwards, giving exactly **120 true / 120 false** from 241
items. A skewed base rate flatters any predictor that leans one way, so this
correction protects the baseline as much as CANDOR.

Final suite: 241 retrieval items, 240 claims (120/120), from a 320-entry sample
at 75% yield. Rejection reasons are recorded in `data/bench/rejects.json`.

---

## Amendment 3 — judge model (2026-07-25)

**Made before any result existed.** The retrieval arm had been computed but not
scored or read; the calibration arm had not run; §4 is untouched.

**Change.** The baseline's probability elicitor and the `agent:llm_big` observer
move from `laguna-s-2.1:q8_0` to `qwen3.6:27b-q8_0`.

**Why.** Measured on this box, with the judge prompt (~1,050 prompt tokens,
~6–9 output tokens):

| model | uncontended | under 4-way concurrency |
|---|---|---|
| `laguna-s-2.1:q8_0` | ~10 s | ~35 s |
| `qwen3.6:27b-q8_0` | ~1 s | — |

The calibration arm needs 240 observer judgements plus 240 elicitations. At
laguna's rate that is several hours of wall clock for one run, and the run has
already been restarted three times for unrelated reasons. This is a hardware
throughput limit, not a modelling choice.

**Why this is not quietly weakening the baseline.** It might be, so it is
measured rather than asserted. `judge_agreement()` in `run_6_8.py` compares the
substituted judge against every laguna judgement still in the cache — on the
same claims, with the same retrieved context and the same prompt — and reports
the agreement rate together with each model's accuracy against ground truth.
That figure ships in `results_6_8.json` under
`calibration.judge_substitution_check` and belongs in any reading of the result.
If the substitute turns out to be a materially *worse* judge, the baseline was
handicapped and the calibration result must be discounted accordingly. Stated
here, in advance, so it cannot be quietly skipped later.

**Note on the original instruction.** `laguna-s-2.1` was nominated for
generating synthetic data. Amendments 1 and 3 together mean it is no longer used
for either role, purely on measured throughput. It remains loaded on the box and
every substitution is recorded here.

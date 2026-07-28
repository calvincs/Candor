"""M8: the guard's held-out validation must be genuinely held out.

The sweep discovers a guard (which covariate, which direction) and then reports
`holdout: {hits, misses}` for the gate's held-out evidence check (gate.py rejects
a guard on hits <= misses). Those hits/misses must be measured on observations
DISJOINT from the ones that drove the discovery — otherwise the "validation" is
just the training set wearing a different hat.

The regression below plants a scenario where the discovery half and the held-out
half disagree, so that a genuine hold-out reports zero misses while the old code
(which re-scored odd-indexed observations of the SAME full set that chose the
guard) is contaminated by discovery-half noise and reports misses > 0.
"""

from __future__ import annotations

from candor.periphery import curiosity_engine as CE


class _FakeIdx:
    """Answers only the three SELECTs curiosity_engine.sweep issues, and records
    (never executes) the UPDATE/INSERT side effects."""

    def __init__(self, fact, obs_rows, ctx_rows):
        self.fact = fact
        self.obs_rows = obs_rows
        self.ctx_rows = ctx_rows
        self.executed: list = []

    def query(self, sql, params=()):
        if "FROM facts f" in sql:
            return [self.fact]
        if "FROM obs_context oc" in sql:
            return list(self.ctx_rows)
        if "FROM observations o" in sql:
            return list(self.obs_rows)
        return []

    def execute(self, sql, params=()):
        self.executed.append((sql, params))


def _plant(n=200):
    """method aliases index parity (even=crawl4ai, odd=http) — the exact shape
    that breaks an even/odd positional split. Discovery-side http is noisy;
    the held-out validation side is clean (crawl4ai always succeeds, http always
    fails), so a genuine hold-out reports zero misses."""
    seqs = list(range(n))
    disc = CE.discovery_mask(seqs)
    obs_rows, ctx_rows = [], []
    for seq, is_disc in zip(seqs, disc):
        method = "crawl4ai" if seq % 2 == 0 else "http"
        if is_disc:
            outcome = True if method == "crawl4ai" else ((seq // 2) % 2 == 0)
        else:
            outcome = method == "crawl4ai"
        obs_rows.append({"event_seq": seq, "outcome": 1 if outcome else 0,
                         "ts": seq})
        ctx_rows.append({"event_seq": seq, "key": "method", "value": method})
    fact = {"id": "f1", "pred": "scrape_ok", "args_json": "[]",
            "stmt_type": "frequency"}
    return fact, obs_rows, ctx_rows


def _buggy_misses(obs_rows, ctx_rows, best="crawl4ai"):
    """What the pre-fix code would have counted: odd-indexed obs of the full set."""
    val = {r["event_seq"]: r["value"] for r in ctx_rows}
    misses = 0
    for i, r in enumerate(obs_rows):
        if i % 2 == 0:
            continue
        predicted = val[r["event_seq"]] == best
        if predicted != bool(r["outcome"]):
            misses += 1
    return misses


def test_guard_holdout_comes_from_the_disjoint_validation_half():
    fact, obs_rows, ctx_rows = _plant()

    # sanity: the construction actually discriminates — the old odd-indexed
    # scoring is contaminated by discovery-half noise and would report misses.
    assert _buggy_misses(obs_rows, ctx_rows) > 0

    idx = _FakeIdx(fact, obs_rows, ctx_rows)
    proposals = CE.sweep(idx)
    guards = [b for kind, b in proposals if kind == "guard"]
    assert len(guards) == 1, "expected exactly one guard proposal"
    body = guards[0]
    assert body["conditioning_key"] == "method"
    assert body["body"]["guards"][0]["value"] == "crawl4ai"

    # the held-out half is clean by construction, so a genuine hold-out reports
    # zero misses; the old full/odd-indexed scoring reported misses > 0.
    assert body["holdout"]["misses"] == 0, (
        f"held-out validation is contaminated by the discovery set: "
        f"{body['holdout']}")
    assert body["holdout"]["hits"] > 0

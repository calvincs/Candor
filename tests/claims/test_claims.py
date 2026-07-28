"""The claims suite: every promise the project makes in public, as a threshold.

Each test names the claim it guards (README.md / docs/*.md) and asserts a
measured number against a bar, on synthetic worlds whose truth is planted. The
point is not that the code runs — the conformance suite covers that — it is
that the *numbers it produces are earned*.

Scale with CLAIMS_SCALE (default 1); CI-sized runs use 1, investigations use
4-8 and take proportionally longer.

  make claims          run everything here
  make claims-fast     drop the slow prediction-heavy tests
"""

from __future__ import annotations

import math
import os
import random

import pytest

from candor.core.committed import reliability as R
from candor.periphery import curiosity_engine as CE
from candor.system import CandorSystem

from . import metrics as M
from . import worlds as W

SCALE = float(os.environ.get("CLAIMS_SCALE", "1"))


def n(base: int) -> int:
    return max(4, int(round(base * SCALE)))


pytestmark = pytest.mark.claims


# ══ helpers ════════════════════════════════════════════════════════════════

def sweep_worlds(streams, pred="svc_ok", stmt_type="frequency"):
    """Feed many independent worlds into one store; return per-world results.

    Each world gets its own fact, so the sweep analyses them independently and
    a single store amortises the setup across all replications.
    """
    m = W.fresh_store("candor-sweep-", actors=["tool:probe"])
    stmts = []
    for i, s in enumerate(streams):
        m.assert_({"pred": pred, "args": [s.label, str(i)], "stmt_type": stmt_type},
                  source="ops", actor="human:me")
        stmts.append({"pred": pred, "args": [s.label, str(i)]})
    m.run_gate()
    fids = []
    for stmt, s in zip(stmts, streams):
        fids.append(m.fact_id_for(stmt))
        W.feed(m, stmt, s)
    proposals = {}
    for kind, body in CE.sweep(m.index):
        proposals.setdefault(body.get("fact_id") or body.get("target_fact"),
                             []).append((kind, body))
    return m, fids, proposals


def changepoint_score(streams) -> M.DetectorScore:
    m, fids, proposals = sweep_worlds(streams)
    fires, loc = 0, []
    for fid, s in zip(fids, streams):
        cps = [b["changepoint_index"] for k, b in proposals.get(fid, [])
               if k == "supersede_valid_time"]
        if cps:
            fires += 1
            if s.changepoint is not None:
                loc.append(abs(cps[0] - s.changepoint))
    m.close()
    return M.DetectorScore(len(streams), fires, loc)


@pytest.fixture(scope="module")
def judge_world():
    """One store, both panels, trained on settled claims then scored held-out.

    Expensive (every held-out claim costs a predict()), so it is built once and
    shared by the trust, composition and calibration tests.
    """
    # The weak panel carries the calibration tests, which need enough held-out
    # mass per reliability bin to be measurable rather than noisy.
    sizes = {"strong": (n(200), n(120)), "weak": (n(150), n(300))}
    panels = {"strong": W.STRONG_PANEL, "weak": W.WEAK_PANEL}
    actors = [j.name for p in panels.values() for j in p] + ["verifier:truth"]
    m = W.fresh_store("candor-judges-", actors=actors)
    m.register_oracle("verifier:truth", "deterministic_total", "gt", "h", "e")
    rng = random.Random(20260727)

    for tag in panels:
        for i in range(sizes[tag][0]):
            m.assert_({"pred": "claim_true", "args": [tag, f"tr{i}"],
                       "stmt_type": "crisp"}, source="suite", actor="human:me")
    m.run_gate()
    for tag, panel in panels.items():
        for i in range(sizes[tag][0]):
            truth = rng.random() < 0.5
            stmt = {"pred": "claim_true", "args": [tag, f"tr{i}"]}
            for j in panel:
                m.observe(stmt, j.vote(truth, rng), {}, actor=j.name)
            m.resolve(m.claim(stmt, frame="external", criterion="verifier:truth",
                              due=0), outcome=truth)

    for tag in panels:
        for i in range(sizes[tag][1]):
            m.assert_({"pred": "claim_true", "args": [tag, f"te{i}"],
                       "stmt_type": "crisp"}, source="suite", actor="human:me")
    m.run_gate()
    held = {tag: [] for tag in panels}
    for tag, panel in panels.items():
        for i in range(sizes[tag][1]):
            truth = rng.random() < 0.5
            stmt = {"pred": "claim_true", "args": [tag, f"te{i}"]}
            votes = {j.name: j.vote(truth, rng) for j in panel}
            for name, v in votes.items():
                m.observe(stmt, v, {}, actor=name)
            held[tag].append((m.predict(stmt, budget=2000).p, truth, votes))
    yield m, held, panels
    m.close()


# ══ CLAIM: "sources earn trust by being right", asymmetrically ═════════════

class TestEarnedTrust:
    """README: 'an always-agreeable source's yes ends up worth nothing, while a
    careful checker's rare no becomes decisive.'"""

    def test_learned_rates_recover_planted_operating_points(self, judge_world):
        """Compared against the PRIOR-SHRUNK truth, not the raw planted rate:
        the Beta(9.5,0.5)/Beta(0.5,9.5) priors deliberately pull a newcomer
        toward 'informative', so a degenerate judge (the always-yes sycophant,
        true fpr 1.0) must land near its shrunk value, not at 1.0. Testing
        against the raw rate would be testing that the prior does not exist."""
        m, _, panels = judge_world
        for j in panels["strong"]:
            tp, fn, fp, tn = R.confusion(m.index, j.name)
            sens, fpr = R.rates((tp, fn, fp, tn))
            exp_sens = (R.SENS_PRIOR[0] + j.sens * (tp + fn)) / (10.0 + tp + fn)
            exp_fpr = (R.FPR_PRIOR[0] + j.fpr * (fp + tn)) / (10.0 + fp + tn)
            assert abs(sens - exp_sens) < 0.06, (
                f"{j.name} sens {sens:.3f}, expected ~{exp_sens:.3f}")
            assert abs(fpr - exp_fpr) < 0.06, (
                f"{j.name} fpr {fpr:.3f}, expected ~{exp_fpr:.3f}")

    def test_sycophants_yes_carries_no_information(self, judge_world):
        m, _, _ = judge_world
        sens, fpr = R.rates(R.confusion(m.index, "agent:sycophant"))
        lr_yes = math.exp(R.log_lr(sens, fpr, True))
        assert lr_yes < 1.3, f"always-yes source's 'yes' still worth {lr_yes:.2f}x"

    def test_rare_but_precise_votes_are_decisive(self, judge_world):
        m, _, _ = judge_world
        s, f = R.rates(R.confusion(m.index, "agent:specialist"))
        assert math.exp(R.log_lr(s, f, True)) > 10, "specialist's rare yes is weak"
        s, f = R.rates(R.confusion(m.index, "agent:alarmist"))
        assert math.exp(R.log_lr(s, f, False)) < 0.15, "alarmist's rare no is weak"


class TestComposition:
    """docs/use-cases.md: 'beat plain vote-averaging by 0.04 Brier'."""

    def test_beats_vote_averaging(self, judge_world):
        _, held, _ = judge_world
        pairs = [(p, t) for p, t, _ in held["strong"]]
        votes = [(sum(v.values()) / len(v), t) for _, t, v in held["strong"]]
        margin = M.brier(votes) - M.brier(pairs)
        assert margin > 0.04, f"only beat vote-averaging by {margin:.4f} Brier"

    def test_approaches_the_independent_judge_ceiling(self, judge_world):
        _, held, panels = judge_world
        pairs = [(p, t) for p, t, _ in held["strong"]]
        oracle = [(M.oracle_nb(v, panels["strong"]), t) for _, t, v in held["strong"]]
        gap = M.brier(pairs) - M.brier(oracle)
        assert gap < 0.015, f"{gap:.4f} Brier short of the oracle ceiling"


class TestCalibration:
    """README: 'when CANDOR says 0.83, that number comes from observed
    outcomes ... never from the embedding was close.'"""

    def test_stated_probabilities_match_observed_frequencies(self, judge_world):
        _, held, _ = judge_world
        pairs = [(p, t) for p, t, _ in held["weak"]]
        assert M.ece(pairs) < 0.05, f"ECE {M.ece(pairs):.4f}"

    def test_not_systematically_over_or_under_confident(self, judge_world):
        _, held, _ = judge_world
        pairs = [(p, t) for p, t, _ in held["weak"]]
        slope = M.calibration_slope(pairs)
        assert 0.85 < slope < 1.15, f"calibration slope {slope:.3f}"

    def test_mean_prediction_tracks_the_base_rate_it_was_never_told(self, judge_world):
        _, held, _ = judge_world
        pairs = [(p, t) for p, t, _ in held["weak"]]
        mp = sum(p for p, _ in pairs) / len(pairs)
        mt = sum(t for _, t in pairs) / len(pairs)
        assert abs(mp - mt) < 0.06, f"mean p {mp:.3f} vs base rate {mt:.3f}"


# ══ CLAIM: instability is a clue — conditions, regimes, honest confusion ═══

class TestRegimeChange:
    """README: 'when something changes for good, it finds the date: regime
    changes are located, not decayed away.'"""

    @pytest.mark.parametrize("before,after", [(0.95, 0.05), (0.90, 0.10)])
    def test_finds_a_tool_that_broke(self, before, after):
        rng = random.Random(hash((before, 11)) & 0xFFFF)
        streams = [W.step(120, 60, before, after, rng) for _ in range(n(40))]
        score = changepoint_score(streams)
        assert score.rate >= 0.90, f"missed {before}->{after} breaks: {score}"

    def test_locates_the_break_precisely(self):
        rng = random.Random(5)
        streams = [W.step(120, 60, 0.9, 0.1, rng) for _ in range(n(40))]
        score = changepoint_score(streams)
        assert score.median_error is not None and score.median_error <= 3, (
            f"changepoint localization degraded: {score}")

    @pytest.mark.parametrize("p", [0.5, 0.9])
    def test_does_not_invent_breaks_in_stationary_streams(self, p):
        rng = random.Random(hash(("flat", p)) & 0xFFFF)
        streams = [W.stationary(120, p, rng) for _ in range(n(40))]
        score = changepoint_score(streams)
        assert score.rate <= 0.05, f"invented a regime change at p={p}: {score}"

    def test_oscillation_is_not_a_regime_change(self):
        rng = random.Random(13)
        streams = [W.oscillating(240, 20, 0.85, 0.35, rng) for _ in range(n(40))]
        score = changepoint_score(streams)
        assert score.rate <= 0.10, f"flapping service read as a break: {score}"

    def test_an_outage_that_recovered_is_not_a_regime_change(self):
        rng = random.Random(17)
        streams = [W.burst(160, 60, 30, 0.9, 0.1, rng) for _ in range(n(40))]
        score = changepoint_score(streams)
        assert score.rate <= 0.15, f"transient outage read as permanent: {score}"

    def test_the_located_date_is_what_gets_committed(self):
        """The whole claim is 'it finds the date'. A located changepoint that
        is discarded on write is not located at all."""
        import time
        m = W.fresh_store("candor-validto-", actors=["tool:probe"])
        stmt = {"pred": "convert_ok", "args": ["svc"]}
        m.assert_({**stmt, "stmt_type": "frequency"}, source="ops", actor="human:me")
        m.run_gate()
        rng = random.Random(3)
        for _ in range(60):
            m.observe(stmt, rng.random() < 0.95, {}, actor="tool:probe")
        time.sleep(1.5)
        boundary = int(time.time() * 1000)
        for _ in range(60):
            m.observe(stmt, rng.random() < 0.05, {}, actor="tool:probe")
        time.sleep(1.5)
        m.run_gate()
        sweep_time = int(time.time() * 1000)
        row = m.index.one("SELECT valid_to FROM facts WHERE id=?",
                          (m.fact_id_for(stmt),))
        m.close()
        assert row["valid_to"], "regime change never reached the committed tier"
        vt = int(row["valid_to"])
        assert abs(vt - boundary) < abs(vt - sweep_time), (
            f"valid_to tracks the sweep ({abs(vt - sweep_time)}ms away), not the "
            f"regime change ({abs(vt - boundary)}ms away)")


class TestConditions:
    """docs/use-cases.md: 'proposes a guard (works when method=crawl4ai), which
    must survive BH correction, an MDL check, and held-out validation.'"""

    def _guards(self, streams):
        m = W.fresh_store("candor-guard-", actors=["tool:probe"])
        stmts = []
        for i, s in enumerate(streams):
            m.assert_({"pred": "scrape_ok", "args": [s.label, str(i)],
                       "stmt_type": "frequency"}, source="ops", actor="human:me")
            stmts.append({"pred": "scrape_ok", "args": [s.label, str(i)]})
        m.run_gate()
        fids = [m.fact_id_for(s) for s in stmts]
        for stmt, s in zip(stmts, streams):
            W.feed(m, stmt, s)
        import json
        m.run_gate()
        found = {}
        for r in m.index.query("SELECT body_json, status FROM candidates "
                               "WHERE kind='guard'"):
            b = json.loads(r["body_json"])
            found.setdefault(b["target_fact"], []).append(
                (b["conditioning_key"], r["status"]))
        return m, fids, found

    @pytest.mark.parametrize("hi,lo,floor", [(0.92, 0.25, 0.95), (0.70, 0.40, 0.85)])
    def test_recovers_the_covariate_that_actually_matters(self, hi, lo, floor):
        rng = random.Random(hash((hi, 3)) & 0xFFFF)
        streams = [W.covariate(200, rng, hi, lo) for _ in range(n(30))]
        m, fids, found = self._guards(streams)
        hits = sum(any(k == "method" and s == "admitted"
                       for k, s in found.get(f, [])) for f in fids)
        m.close()
        assert hits / len(fids) >= floor, (
            f"recovered method in only {hits}/{len(fids)} streams")

    def test_does_not_manufacture_conditions_from_noise(self):
        """Five pure-noise covariates per stream and no real effect: any guard
        admitted here is a fishing expedition that survived."""
        rng = random.Random(29)
        streams = [W.covariate(200, rng, 0.6, 0.6) for _ in range(n(30))]
        m, fids, found = self._guards(streams)
        bad = sum(any(s == "admitted" for _, s in found.get(f, [])) for f in fids)
        m.close()
        assert bad / len(fids) <= 0.08, (
            f"{bad}/{len(fids)} streams got a guard from pure noise")


class TestHonestConfusion:
    """docs/use-cases.md: 'unexplained variance — opens a question carrying the
    residual partition and a concrete suggested measurement.'"""

    @pytest.mark.parametrize("ctx_fn,label", [
        (lambda i: {}, "nothing logged"),
        (lambda i: {"noise": "a" if i % 3 else "b"}, "only irrelevant context"),
    ])
    def test_says_something_when_it_cannot_explain_the_variance(self, ctx_fn, label):
        """An agent whose tool swings between 85% and 35% must be TOLD, even
        when it logged nothing that explains why. Silence is the failure mode
        this project exists to prevent."""
        rng = random.Random(41)
        streams = [W.oscillating(240, 20, 0.85, 0.35, rng) for _ in range(n(20))]
        m = W.fresh_store("candor-silent-", actors=["tool:probe"])
        stmts = []
        for i, s in enumerate(streams):
            m.assert_({"pred": "flaky_ok", "args": [str(i)],
                       "stmt_type": "frequency"}, source="ops", actor="human:me")
            stmts.append({"pred": "flaky_ok", "args": [str(i)]})
        m.run_gate()
        fids = [m.fact_id_for(s) for s in stmts]
        for stmt, s in zip(stmts, streams):
            W.feed(m, stmt, s, ctx_fn=ctx_fn)
        m.run_gate()
        asked = {q["target_id"] for q in m.questions()}
        flagged = {r["id"] for r in m.index.query(
            "SELECT id FROM facts WHERE dispersion_flag=1")}
        told = sum((f in asked) or (f in flagged) for f in fids)
        m.close()
        assert told / len(fids) >= 0.80, (
            f"with {label}, only {told}/{len(fids)} unstable streams produced "
            f"any signal at all")

    def test_a_question_names_a_measurement_to_take(self):
        rng = random.Random(43)
        streams = [W.oscillating(240, 20, 0.85, 0.35, rng) for _ in range(n(8))]
        m = W.fresh_store("candor-ask-", actors=["tool:probe"])
        stmts = []
        for i, s in enumerate(streams):
            m.assert_({"pred": "flaky2_ok", "args": [str(i)],
                       "stmt_type": "frequency"}, source="ops", actor="human:me")
            stmts.append({"pred": "flaky2_ok", "args": [str(i)]})
        m.run_gate()
        for stmt, s in zip(stmts, streams):
            W.feed(m, stmt, s, ctx_fn=lambda i: {})
        m.run_gate()
        qs = m.questions()
        m.close()
        assert qs, "no question opened for wholly unexplained instability"
        assert all(q["suggested_measurement"] for q in qs), (
            "a question with no suggested measurement is just a shrug")


# ══ CLAIM: nothing is silently mutated; retraction is total ════════════════

class TestProvenance:
    """README: 'Delete the SQLite index entirely and rebuild it bit-for-bit
    from the log. Retract a poisoned source and every downstream number
    recomputes as if it never spoke.'"""

    GOOD = ["tool:probe-a", "tool:probe-b"]
    POISON = "tool:bad-scraper"

    def _build(self, root, poison: bool, shuffle: bool = False):
        m = CandorSystem(root)
        for a in self.GOOD + [self.POISON, "human:me"]:
            m.set_actor_quota(a, obs_per_epoch=W.BIG_QUOTA, cand_per_epoch=W.BIG_QUOTA)
        for i in range(12):
            m.assert_({"pred": "holds", "args": [f"t{i}"], "stmt_type": "crisp"},
                      source="suite", actor="human:me")
        m.run_gate()
        rng = random.Random(1)
        events = []
        for i in range(12):
            stmt = {"pred": "holds", "args": [f"t{i}"]}
            for _ in range(10):
                for a in self.GOOD:
                    events.append((stmt, rng.random() < 0.7, a))
                if poison:
                    events.append((stmt, False, self.POISON))
        if shuffle:
            random.Random(99).shuffle(events)
        for stmt, out, a in events:
            m.observe(stmt, out, {}, actor=a)
        m.run_gate()
        return m

    def _preds(self, m):
        return {i: repr(m.predict({"pred": "holds", "args": [f"t{i}"]},
                                  budget=1500).p) for i in range(12)}

    def test_index_rebuilds_from_the_log(self, tmp_path):
        m = self._build(tmp_path / "a", poison=True)
        before, closure, head = self._preds(m), m.closure_hash(), m.ledger_head()
        m.close()
        (tmp_path / "a" / "index.sqlite3").unlink()
        m2 = CandorSystem(tmp_path / "a")
        after, closure2, head2 = self._preds(m2), m2.closure_hash(), m2.ledger_head()
        m2.close()
        assert after == before, "predictions changed after rebuild"
        assert (closure2, head2) == (closure, head), "derived hashes changed"

    def test_predictions_are_independent_of_insertion_order(self, tmp_path):
        a = self._build(tmp_path / "a", poison=True)
        ordered = self._preds(a)
        a.close()
        b = self._build(tmp_path / "b", poison=True, shuffle=True)
        shuffled = self._preds(b)
        b.close()
        assert ordered == shuffled

    def test_a_recorded_snapshot_reproduces_its_number(self, tmp_path):
        m = self._build(tmp_path / "a", poison=True)
        stmt = {"pred": "holds", "args": ["t0"]}
        p1 = m.predict(stmt, budget=1500)
        m.observe(stmt, False, {}, actor=self.GOOD[0])
        m.observe(stmt, False, {}, actor=self.GOOD[1])
        moved = m.predict(stmt, budget=1500).p
        back = m.predict_at(stmt, p1.snapshot_id).p
        m.close()
        assert moved != p1.p, "the world did not actually move"
        assert back == p1.p, f"snapshot replay drifted: {back!r} != {p1.p!r}"

    def test_retracting_a_source_leaves_other_actors_untouched(self, tmp_path):
        """The documented recovery from a hallucinating scraper. Purging it
        must not take honest observations with it — and identical reports from
        different actors are the NORMAL case, not a corner case."""
        m = self._build(tmp_path / "p", poison=True)
        rows = m.index.query(
            "SELECT payload_hash, actor FROM events WHERE kind='observation'")
        honest_before = sum(1 for r in rows if r["actor"] != self.POISON)
        m.retract_source(self.POISON, reason="hallucinating scraper")
        m.replay()
        honest_after = m.index.one(
            "SELECT COUNT(*) n FROM observations WHERE actor != ?",
            (self.POISON,))["n"]
        poison_after = m.index.one(
            "SELECT COUNT(*) n FROM observations WHERE actor = ?",
            (self.POISON,))["n"]
        m.close()
        assert poison_after == 0, "retracted source still has live observations"
        assert honest_after == honest_before, (
            f"retraction destroyed {honest_before - honest_after} honest "
            f"observations belonging to other actors")

    def test_retraction_reproduces_a_store_the_source_never_touched(self, tmp_path):
        clean = self._build(tmp_path / "clean", poison=False)
        expected = self._preds(clean)
        clean.close()
        m = self._build(tmp_path / "dirty", poison=True)
        poisoned = self._preds(m)
        m.retract_source(self.POISON, reason="hallucinating scraper")
        m.replay()
        recovered = self._preds(m)
        m.close()
        assert poisoned != expected, "the poison never had an effect to undo"
        assert recovered == expected, "numbers did not recompute as if it never spoke"

    def test_discounting_a_source_moves_crisp_beliefs(self, tmp_path):
        """docs/use-cases.md offers set_reliability as the softer scalpel. A
        lever the docs hand the operator must actually move the thing it names."""
        clean = self._build(tmp_path / "clean", poison=False)
        expected = self._preds(clean)
        clean.close()
        m = self._build(tmp_path / "dirty", poison=True)
        before = self._preds(m)
        m.set_reliability(self.POISON, "external", 0.001, 100)
        after = self._preds(m)
        m.close()
        assert after != before, "set_reliability() did not move a crisp belief"
        moved = sum(abs(float(after[i]) - float(expected[i]))
                    < abs(float(before[i]) - float(expected[i])) for i in range(12))
        assert moved >= 10, (
            f"discounting moved only {moved}/12 beliefs toward the clean value")

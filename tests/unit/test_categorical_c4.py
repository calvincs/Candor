"""Categorical facts — Stage C4: curiosity / context-conditioning + the predict
context breakdown (design_categorical.md §7 + §3 + §5/§6, "Stage C4"). The FINAL
categorical stage: it closes the loop from "which value" to "which value, WHEN".

C4 delivers three things, all strictly additive to C1–C3:

  * PER-VALUE ONE-VS-REST SWEEP (§7, LOCKED to one-vs-rest, not the deferred joint
    G-test) — for each observed value v the categorical series is projected to the
    binary series [value == v] and the EXISTING per-fact sweep (partition_by_key →
    Tarone → BH → MDL → held-out → guard) runs UNCHANGED on each projection, with
    Benjamini–Hochberg correcting across the K-values × keys comparisons together.
    So "region=eu ⇒ captcha" surfaces as a guard conditioning the value on region,
    proposed and admitted through the ordinary gate flow. A NULL world (value
    independent of every key) admits no spurious guard (the BH/holdout control).
  * PREDICT CONTEXT BREAKDOWN (§3) — CategoricalPrediction.by_context conditions
    the value distribution on a discovered key: each context group gets its OWN
    CRP predictive (its own unknown slice), plus a __residual__ group for obs that
    did not record the key, reusing Feature A's outcome_breakdown.
  * BREADTH over ALL observations' context entropy (§7) — every categorical obs is
    informative; the binary "confirming-observations" breadth is untouched.

  DETERMINISM (I3/I8) — the categorical sweep is a pure function of the fact's own
  observations; replay reproduces closure_hash, predict_at reproduces by_context,
  checkpoint == full replay. ADDITIVITY (§6) — the binary sweep/breadth and the
  crisp/frequency + Feature-A read paths are behavior-unchanged.
"""

from __future__ import annotations

import json
import random

from candor.core.committed import counts as counts_mod
from candor.periphery import curiosity as C
from candor.periphery import curiosity_engine as CE
from candor.system import CandorSystem, CategoricalPrediction

CAT = {"pred": "resolves", "args": ["login"], "stmt_type": "categorical"}
STMT = {"pred": "resolves", "args": ["login"]}
CRISP = {"pred": "link_ok", "args": ["a", "b"], "stmt_type": "crisp"}
BUDGET = 1000


# ── fixtures / helpers ─────────────────────────────────────────────────────────

def _categorical_store(root):
    m = CandorSystem(root)
    m.set_actor_quota("tool:probe", obs_per_epoch=1_000_000, cand_per_epoch=1_000_000)
    m.assert_(CAT, source="seed", actor="tool:probe")
    m.run_gate()
    return m


def _plant_conditioned(m, seed=7, residual=0):
    """region=eu ⇒ mostly 'captcha', region=us ⇒ mostly 'full-page' (value is
    conditioned on region). Optionally add `residual` region-less observations —
    the honest 'cannot attribute' mass that lands in the __residual__ bucket."""
    rng = random.Random(seed)
    plan = []
    for _ in range(70):
        plan.append(("eu", "captcha" if rng.random() < 0.9 else "block"))
    for _ in range(70):
        plan.append(("us", "full-page" if rng.random() < 0.9 else "block"))
    rng.shuffle(plan)
    for region, value in plan:
        m.observe(STMT, ctx={"region": region}, actor="tool:probe", value=value)
    for i in range(residual):
        m.observe(STMT, ctx={}, actor="tool:probe",
                  value="captcha" if i % 2 else "full-page")
    m.run_gate()


def _plant_null(m, seed=11):
    """region records a value on every observation but the VALUE is independent of
    it — every context has the same mixed distribution. No key conditions it."""
    rng = random.Random(seed)
    vocab = ["captcha", "full-page", "block"]
    plan = [("eu", rng.choice(vocab)) for _ in range(70)]
    plan += [("us", rng.choice(vocab)) for _ in range(70)]
    rng.shuffle(plan)
    for region, value in plan:
        m.observe(STMT, ctx={"region": region}, actor="tool:probe", value=value)
    m.run_gate()


def _admitted_guards(m, fid):
    out = []
    for r in m.index.query("SELECT body_json FROM candidates WHERE kind='guard' "
                           "AND status='admitted' ORDER BY event_seq"):
        body = json.loads(r["body_json"])
        if body.get("target_fact") == fid:
            out.append(body)
    return out


# ── pure helper (fast, no store) ────────────────────────────────────────────────

def test_category_group_posterior_is_a_proper_crp_predictive_with_its_own_unknown():
    """One context group's CRP predictive: P(v)=n_v/(N+alpha), unknown=alpha/(N+a),
    Σ P(v) + unknown == 1 exactly, and a thin group is mostly-unknown (§3)."""
    post = counts_mod.category_group_posterior(
        {"captcha": 9, "block": 1}, total_n=10, alpha=1.0)
    assert abs(post.values["captcha"].p - 9 / 11) < 1e-12
    assert abs(post.values["block"].p - 1 / 11) < 1e-12
    assert abs(post.unknown.p - 1 / 11) < 1e-12
    total = sum(s.p for s in post.values.values()) + post.unknown.p
    assert abs(total - 1.0) < 1e-12
    # thin group ⇒ large unknown; empty group ⇒ all unknown
    thin = counts_mod.category_group_posterior({"x": 1}, 1, 1.0)
    assert thin.unknown.p == 0.5
    empty = counts_mod.category_group_posterior({}, 0, 1.0)
    assert empty.unknown.p == 1.0 and empty.values == {}


# ══ CONDITIONING DISCOVERED — the seeded world admits, the null world does not ═══

def test_seeded_world_admits_a_guard_conditioning_the_value_on_region(tmp_path):
    m = _categorical_store(tmp_path / "cond")
    fid = m.fact_id_for(STMT)
    _plant_conditioned(m)

    guards = _admitted_guards(m, fid)
    assert len(guards) == 1, "the per-value one-vs-rest sweep must admit one guard"
    g = guards[0]
    # the guard conditions the VALUE on region (design §7): its variable is region,
    # the conditioned value is one of the concentrating values, at its context.
    assert g["conditioning_key"] == "region"
    assert g["conditioned_value"] in ("captcha", "full-page")
    assert g["body"]["guards"][0]["var"] == "?region"
    assert g["body"]["guards"][0]["value"] in ("eu", "us")
    # the held-out check (disjoint validation quarter) is decisively positive
    assert g["holdout"]["hits"] > g["holdout"]["misses"]
    assert bool(m.index.one("SELECT dispersion_flag FROM facts WHERE id=?",
                            (fid,))["dispersion_flag"])
    m.close()


def test_null_world_admits_no_spurious_guard(tmp_path):
    """False-positive control: the value is independent of every context key, so
    across the K-values × keys comparisons BH admits nothing (§7)."""
    m = _categorical_store(tmp_path / "null")
    fid = m.fact_id_for(STMT)
    _plant_null(m)
    assert _admitted_guards(m, fid) == []
    assert not bool(m.index.one("SELECT dispersion_flag FROM facts WHERE id=?",
                                (fid,))["dispersion_flag"])
    m.close()


# ══ CONTEXT BREAKDOWN — predict surfaces the conditioned value distributions ═════

def test_predict_surfaces_per_context_value_distributions_with_unknown_and_residual(tmp_path):
    m = _categorical_store(tmp_path / "cond")
    _plant_conditioned(m, residual=15)     # 15 region-less obs → a residual bucket

    p = m.predict(STMT, budget=BUDGET)
    assert isinstance(p, CategoricalPrediction)
    assert "region" in p.by_context, "the discovered conditioning key must surface"
    region = p.by_context["region"]
    assert set(region) == {"eu", "us", C.RESIDUAL_BUCKET}

    eu, us, res = region["eu"], region["us"], region[C.RESIDUAL_BUCKET]
    # eu concentrates on captcha, us on full-page — the conditioning is legible
    assert max(eu["values"], key=lambda v: eu["values"][v].p) == "captcha"
    assert max(us["values"], key=lambda v: us["values"][v].p) == "full-page"
    assert eu["values"]["captcha"].p > 0.8 and us["values"]["full-page"].p > 0.8
    # each context group carries its OWN unknown slice (its own CRP predictive)
    for grp in (eu, us, res):
        total = sum(s.p for s in grp["values"].values()) + grp["unknown"].p
        assert abs(total - 1.0) < 1e-12
        assert 0.0 < grp["unknown"].p < 1.0
    # the residual group is the honest 'cannot attribute' mass, kept distinct
    assert res["n"] == 15
    # the marginal is untouched and still sums to 1 (additivity with C2)
    assert abs(sum(s.p for s in p.values.values()) + p.unknown.p - 1.0) < 1e-12
    m.close()


def test_recorded_context_but_no_guard_is_under_specified_and_by_context_empty(tmp_path):
    """The null world records region but nothing conditions on it: the marginal is
    honest but under-specified (§3), and by_context stays empty."""
    m = _categorical_store(tmp_path / "null")
    _plant_null(m)
    p = m.predict(STMT, budget=BUDGET)
    assert p.by_context == {}
    assert "under_specified" in p.caveats
    m.close()


def test_no_recorded_context_leaves_by_context_empty_and_no_under_specified(tmp_path):
    """A categorical fact observed with no context keys: no by_context, and NO
    under_specified caveat (there is nothing to condition on) — additivity with the
    C2 predict, whose stores all observe with ctx={}."""
    m = _categorical_store(tmp_path / "plain")
    for v in ["captcha"] * 8 + ["block"] * 4 + ["allow"] * 4:
        m.observe(STMT, ctx={}, actor="tool:probe", value=v)
    m.run_gate()
    p = m.predict(STMT, budget=BUDGET)
    assert p.by_context == {}
    assert "under_specified" not in p.caveats
    m.close()


# ══ DETERMINISM (I3 / I8) ═══════════════════════════════════════════════════════

def test_categorical_sweep_is_a_pure_function_of_the_facts_observations(tmp_path):
    """Re-running the sweep, and re-deriving the ONE fact via resweep, both yield
    byte-identical guard proposals — the resweep purity contract for categorical."""
    m = _categorical_store(tmp_path / "cond")
    fid = m.fact_id_for(STMT)
    _plant_conditioned(m)

    def cat_guards(props):
        return sorted(json.dumps(b, sort_keys=True) for k, b in props
                      if k == "guard" and b.get("target_fact") == fid)

    twice = cat_guards(CE.sweep(m.index))
    again = cat_guards(CE.sweep(m.index))
    assert twice == again and len(twice) == 1
    # resweep of just this fact (the checkpoint fast path) re-derives the same
    reswept = cat_guards(CE.resweep(m.index, [fid]))
    assert reswept == twice
    m.close()


def test_replay_reproduces_closure_hash_with_a_conditioned_categorical_fact(tmp_path):
    m = _categorical_store(tmp_path / "cond")
    _plant_conditioned(m, residual=15)
    before = m.closure_hash()
    assert m.replay() == before, "categorical conditioning must survive replay"
    m.close()


def test_predict_at_reproduces_the_by_context_breakdown_bit_for_bit(tmp_path):
    m = _categorical_store(tmp_path / "cond")
    _plant_conditioned(m, residual=15)
    live = m.predict(STMT, budget=BUDGET)
    snap = live.snapshot_id
    # advance the head past the snapshot (incl. a brand-new value)
    m.observe(STMT, ctx={"region": "eu"}, actor="tool:probe", value="tail-x")
    m.run_gate()
    assert m.ledger.head() != snap

    at = m.predict_at(STMT, snap)
    assert at.by_context == live.by_context      # bit-for-bit, incl. residual
    assert at.values == live.values and at.unknown == live.unknown
    assert at.snapshot_id == live.snapshot_id
    m.close()


def test_checkpoint_accelerated_open_equals_full_replay(tmp_path):
    root = tmp_path / "store"
    m = _categorical_store(root)
    _plant_conditioned(m, residual=15)
    m.checkpoint()
    m.observe(STMT, ctx={"region": "us"}, actor="tool:probe", value="full-page")
    m.run_gate()
    m.close()

    m2 = CandorSystem(root)                          # fast-path open (post-sweep cp)
    hash_cp = m2.closure_hash()
    hash_full = m2.replay()                          # forced full-from-genesis
    assert hash_cp == hash_full
    m2.close()


# ══ ADDITIVITY (§6) — binary sweep / breadth / distribution untouched ════════════

def test_binary_guard_and_breadth_still_work_alongside_categorical(tmp_path):
    """The binary sweep path is byte-identical: a crisp fact with a genuinely
    conditioning covariate still admits its guard and gets a breadth class, even in
    a store that also carries a swept categorical fact."""
    m = _categorical_store(tmp_path / "mixed")
    _plant_conditioned(m)                            # categorical, swept per-value
    m.assert_(CRISP, source="seed", actor="tool:probe")
    m.run_gate()
    # region genuinely predictive for the binary fact: us-east ~0.9, eu ~0.28
    obs = ([(CRISP, i < 36, {"region": "us-east"}, "tool:probe") for i in range(40)]
           + [(CRISP, i < 7, {"region": "eu"}, "tool:probe") for i in range(25)]
           + [(CRISP, i % 2 == 0, {}, "tool:probe") for i in range(17)])
    random.Random(7).shuffle(obs)
    m.observe_batch(obs)
    m.run_gate()

    cfid = m.fact_id_for(CRISP)
    bguards = _admitted_guards(m, cfid)
    assert len(bguards) == 1 and bguards[0]["conditioning_key"] == "region"
    # binary breakdown (Feature A) is unchanged and still binary-shaped
    d = m.distribution(CRISP)
    assert d["found"] and d["stmt_type"] == "crisp" and d["flaky"]
    assert d["residual"]["conditioning_key"] == "region"
    # a binary fact never grows a by_context field (that lives on categorical only)
    assert not hasattr(m.predict(CRISP, budget=BUDGET), "by_context")
    m.close()

"""Curiosity sweep — wiring the §4.5 statistics to the candidate boundary.

Untrusted proposer (I10): everything found here becomes a *candidate* asserted
by `agent:curiosity` and judged by the gate like any other. The sweep is a
deterministic function of the observation log, so replay + sweep reproduces
flags and questions bit-for-bit; while a pattern persists undetected-repaired
it is re-proposed each sweep (churn accepted for v1, absorbed by gate
idempotence).

Routing per §4.5: a time-step function with no explaining covariate is a
regime change (supersede-with-valid-time); a time-stable covariate split that
clears support + MDL + BH is a guard. Time is always tested, with its own
machinery (CUSUM), never as partition-by-bins.
"""

from __future__ import annotations

import math
from typing import Any

from . import curiosity as C

MIN_OBS = 16


def _avalanche(x: int) -> int:
    """SplitMix64 finalizer: a deterministic, replay-stable 64-bit hash whose
    output bits are decorrelated from the input's low-order structure."""
    mask = (1 << 64) - 1
    x &= mask
    x = ((x ^ (x >> 30)) * 0xBF58476D1CE4E5B9) & mask
    x = ((x ^ (x >> 27)) * 0x94D049BB133111EB) & mask
    return x ^ (x >> 31)


#: One in every DISCOVERY_STRIDE observations (in hash-shuffled order) is routed
#: to the held-out VALIDATION set; the rest drive DISCOVERY. Not one-half: a
#: 50/50 split halves the effective sample on each side, which (i) drops a weak
#: but real covariate (0.70 vs 0.40 success) below the detection bar and (ii)
#: pushes a two-value partition below the gate's 8-per-side support floor on
#: short streams. A 3:1 split keeps discovery powerful and every partition clear
#: of the floor while leaving a validation quarter large enough to reject a
#: mis-selected direction.
DISCOVERY_STRIDE = 4


def discovery_mask(seqs: "list[int]") -> list[bool]:
    """A deterministic discovery/validation split aligned with `seqs`: True marks
    a discovery observation, False a held-out validation one.

    The split is *systematic* over a hash-shuffled order — sort the observations
    by a hash of their immutable event_seq, then hand every DISCOVERY_STRIDE-th
    one to validation. That buys three properties a plain positional even/odd
    split lacks:

    * Decorrelation. A covariate recorded as ``value[i % 2]`` — an alternating
      method, a flip-flopping route — aliases exactly with index parity, so an
      'even-indexed' discovery set would see only one covariate value and no
      contrast at all. Sorting by a hash of the event_seq breaks that aliasing.
    * Balance. Systematic sampling of the shuffled order splits every covariate
      value into discovery/validation with far lower allocation variance than
      independent per-observation hashing, which matters when a covariate is
      recorded on only a few dozen observations (the direction stays estimable).
    * Replay-stability. Keyed on the immutable event_seq, the split is
      bit-for-bit reproducible on replay.
    """
    order = sorted(range(len(seqs)), key=lambda i: (_avalanche(seqs[i]), seqs[i]))
    mask = [True] * len(seqs)
    for rank, i in enumerate(order):
        if rank % DISCOVERY_STRIDE == DISCOVERY_STRIDE - 1:
            mask[i] = False
    return mask


def _mdl_fields(groups: dict[str, C.Group]) -> dict[str, float]:
    pooled = [C.Group(sum(g.n for g in groups.values()),
                      sum(g.k for g in groups.values()))]
    split = list(groups.values())
    bits = 2.0 + math.log2(max(2, len(groups)))          # guard description cost
    return C.mdl_gain(pooled, split, guard_bits=bits)


def sweep(idx) -> list[tuple[str, dict[str, Any]]]:
    """FULL sweep: (re)derive every fact's verdict from its own observations.

    Returns (candidate_kind, body) proposals; also sets each fact's dispersion
    flag / breadth / open question. A fact's verdict is a pure function of that
    fact's OWN observations and context (nothing here reads across facts), which
    is exactly what lets `resweep` re-derive one fact in isolation and match this.
    """
    proposals: list[tuple[str, dict[str, Any]]] = []
    # The sweep is a memoryless detector over the raw observation log: a
    # persisting pattern is re-proposed every sweep (gate application is
    # idempotent). Filtering repaired facts here would make detection depend
    # on repair history and break replay-equivalence of the sweep.
    facts = idx.query(
        "SELECT f.id, f.pred, f.args_json, f.stmt_type FROM facts f "
        "ORDER BY f.id")
    for fact in facts:
        proposals.extend(_sweep_fact(idx, fact))
    return proposals


def resweep(idx, fact_ids) -> list[tuple[str, dict[str, Any]]]:
    """INCREMENTAL sweep: re-derive ONLY the given facts' verdicts (H6b).

    On the checkpoint fast path the restored snapshot is POST-sweep, so a fact's
    verdict is already correct UNLESS the folded tail added an observation to it.
    Re-deriving just those tail-touched facts — each from its FULL observation set
    — is byte-identical to a full sweep for them (a verdict depends only on the
    fact's own observations), while untouched facts keep their snapshot verdict.
    O(touched), not O(all facts).

    Because a re-derivation must be able to DOWNGRADE a fact (the sweep only ever
    SETS dispersion_flag and always (re)writes breadth), each fact's prior verdict
    is wiped first, then recomputed. On a from-scratch full replay the same facts
    start blank, so `sweep` needs no such wipe and stays byte-for-byte as before.
    """
    proposals: list[tuple[str, dict[str, Any]]] = []
    for fid in fact_ids:
        fact = idx.one(
            "SELECT id, pred, args_json, stmt_type FROM facts WHERE id=?", (fid,))
        if fact is None:
            continue
        idx.execute(
            "UPDATE facts SET dispersion_flag=0, breadth_class=NULL WHERE id=?",
            (fid,))
        idx.execute(
            "DELETE FROM open_questions WHERE kind='dispersion' AND target_id=?",
            (fid,))
        proposals.extend(_sweep_fact(idx, fact))
    return proposals


def _sweep_fact(idx, fact) -> list[tuple[str, dict[str, Any]]]:
    """Derive one fact's verdict from its full observation set. Pure in the
    fact's own observations/context — the unit shared by `sweep` and `resweep`."""
    if fact["stmt_type"] == "categorical":
        # A categorical fact carries no boolean outcome, so the binomial sweep
        # below does not apply; it is swept per-value one-vs-rest (C4, design §7).
        # Kept a disjoint branch so the crisp/frequency path stays byte-identical.
        return _sweep_categorical_fact(idx, fact)
    proposals: list[tuple[str, dict[str, Any]]] = []
    rows = idx.query(
        "SELECT o.event_seq, o.outcome, e.ts FROM observations o "
        "JOIN events e ON e.seq = o.event_seq WHERE o.fact_id=? "
        "ORDER BY o.event_seq", (fact["id"],))
    if len(rows) >= MIN_OBS:
        series = [bool(r["outcome"]) for r in rows]
        ctx_rows = idx.query(
            "SELECT oc.event_seq, oc.key, oc.value FROM obs_context oc "
            "JOIN observations o ON o.event_seq = oc.event_seq "
            "WHERE o.fact_id=?", (fact["id"],))
        by_seq: dict[int, dict[str, str]] = {}
        for r in ctx_rows:
            by_seq.setdefault(int(r["event_seq"]), {})[r["key"]] = r["value"]
        # Δ10: the covariate search runs over the RECORDED context augmented with
        # synthesized frames (hour/dow from the event ts, the fact's own previous
        # outcome, pairwise interactions of recorded keys). A pure per-fact
        # function of the log, so resweep purity and replay hold. `base_ctx` (the
        # recorded truth) is kept for breadth, which measures the AGENT's logging
        # diversity — a synthesized hour must never inflate it.
        base_ctx = [by_seq.get(int(r["event_seq"]), {}) for r in rows]
        prevs: list = [None] + ["T" if bool(rows[i - 1]["outcome"]) else "F"
                                for i in range(1, len(rows))]
        aug = C.augment_derived(base_ctx, [int(r["ts"]) for r in rows], prevs)
        obs = [(aug[i], bool(r["outcome"])) for i, r in enumerate(rows)]
        keys = sorted({k for ctx, _ in obs for k in ctx})

        # §3.4 step 5: a GENUINE discovery/validation split. Every discovery
        # statistic below (Tarone / BH / MDL / direction) is computed on the
        # discovery observations only; the hits/misses handed to the gate come
        # from the disjoint, untouched validation quarter. Otherwise the gate's
        # held-out check (gate.py rejects on hits <= misses) runs on data that
        # already drove the selection, which is no check at all (M8).
        seqs = [int(r["event_seq"]) for r in rows]
        disc = discovery_mask(seqs)
        disc_obs = [o for o, d in zip(obs, disc) if d]
        val_obs = [o for o, d in zip(obs, disc) if not d]

        # covariate search: Tarone per key, BH across the keys tested (§4.5).
        # A key whose cardinality exceeds n/(2*min_support) cannot yield a
        # guard — an m-ary split at that granularity is a lookup table, not a
        # condition — but it still licenses DETECTION (flag + open question).
        tested: list[tuple[str, dict[str, C.Group], float, bool]] = []
        for key in keys:
            groups = C.partition_by_key(disc_obs, key)
            usable = {v: g for v, g in groups.items()
                      if g.n >= C.MIN_SUPPORT_PER_PARTITION}
            if len(usable) < 2:
                continue
            z = C.tarone_z(list(usable.values()))
            if z is None:
                continue
            # M9: a far-tail-calibrated p-value, not the anti-conservative normal
            # approximation, so the BH threshold this feeds holds its nominal
            # false-guard rate out where alpha/m bites.
            pvalue = C.tarone_pvalue(list(usable.values()))
            if pvalue is None:
                continue
            guardable = len(groups) <= max(
                2, len(disc_obs) // (2 * C.MIN_SUPPORT_PER_PARTITION))
            tested.append((key, usable, pvalue, guardable))
        # §4.4 routing: a regime change is ONE-WAY. Locate the shift at the
        # argmax of cumulative deviation, test it exactly against the search
        # that found it, then ask whether either side changes AGAIN — if so the
        # series oscillates, which is dispersion wearing a changepoint costume,
        # and the repair is a condition or a question, never a supersede.
        # (Computed BEFORE winner selection: Δ10's derived:prev must yield to it.)
        detected = C.changepoint_test(series)
        changepoint = detected[0] if detected else None
        changepoint_p = detected[1] if detected else None
        recurrent = C.is_recurrent(series) if detected else False
        one_way = changepoint is not None and not recurrent

        keep = C.benjamini_hochberg([t[2] for t in tested]) if tested else []
        winner = None
        # Δ10: recorded keys outrank derived ones at winner selection — the
        # agent's own vocabulary is the primary explanation space; a synthesized
        # frame speaks only when nothing the agent logged explains the variance.
        ranked = sorted(zip(keep, tested),
                        key=lambda ft: (C.is_derived(ft[1][0]), ft[1][0]))
        for flag, (key, usable, _, guardable) in ranked:
            if not (flag and guardable):
                continue
            if key == C.DERIVED_PREV:
                # Self-lag is the frame most prone to shadowing OTHER structure:
                # a one-way step makes prev≈current by construction (the honest
                # repair is the located DATE, §4.4), and an unlogged block
                # variable makes prev a smeared proxy (the honest repair is an
                # open question saying "log wider"). prev may speak only when it
                # ABSORBS the time structure it claims to explain: neither
                # residual subseries may still carry temporal dispersion or a
                # leftover step. A genuine sticky process passes (conditioned on
                # prev, it is stationary); the shadows fail.
                if one_way or not _prev_absorbs_time(obs):
                    continue
            mdl = _mdl_fields(usable)
            if mdl["dl_guard"] + mdl["dl_residual_given_guard"] < mdl["dl_residual"]:
                winner = (key, usable, mdl)
                break

        if winner is not None:
            key, usable, mdl = winner
            best = max(usable, key=lambda v: usable[v].k / max(1, usable[v].n))
            # The guard and its direction were chosen on the discovery
            # observations; the hits/misses the gate weighs are scored on the
            # disjoint validation set, which no discovery statistic touched (M8).
            hits = misses = 0
            for ctx, out in val_obs:
                if key not in ctx:
                    continue
                predicted = ctx[key] == best
                if predicted == out:
                    hits += 1
                else:
                    misses += 1
            proposal = {
                "head": {"pred": fact["pred"], "args": ["?x"]},
                "body": {"literals": [],
                         "guards": [{"var": f"?{key}", "op": "==", "value": best}]},
                "support": {"left": min(g.n for g in usable.values()),
                            "right": max(g.n for g in usable.values())},
                "holdout": {"hits": hits, "misses": misses},
                "mdl": mdl, "specificity": 1, "conditioning_key": key,
                "target_fact": fact["id"],
            }
            if C.is_do(key):
                # Δ13: conditioning on an intervention key is not a mere
                # condition — it says the coupling is REGIME-dependent.
                proposal["regime_dependent"] = True
            proposals.append(("guard", proposal))
            idx.execute("UPDATE facts SET dispersion_flag=1 WHERE id=?",
                        (fact["id"],))
            _open_question(idx, fact["id"], usable, key)
        elif changepoint is not None and not recurrent:
            # The whole point of locating a changepoint is to record WHEN the
            # old regime stopped holding. Carry the located observation's own
            # timestamp, or the commit stamps the sweep's wall clock instead
            # and the located date is thrown away (F3).
            proposals.append(("supersede_valid_time", {
                "fact_id": fact["id"], "changepoint_index": changepoint,
                "valid_to": int(rows[changepoint]["ts"]),
                "changepoint_event_seq": int(rows[changepoint]["event_seq"]),
                "support": {"before": changepoint + 1,
                            "after": len(series) - changepoint - 1},
                "pvalue": changepoint_p,
                "reason": "one-way level change, no explaining covariate",
            }))
        # Under-explained instability: nothing was guarded and no single date
        # accounts for it. Two independent grounds to speak, either sufficient —
        # a recorded covariate that clusters the variance without clearing the
        # guard bar, or dispersion on the time axis itself, which needs no
        # covariate at all. Requiring the former was the whole defect: an agent
        # that logged nothing relevant was told nothing (F5).
        if winner is None and (changepoint is None or recurrent):
            z_any = max((C.tarone_z(list(u.values())) or 0.0)
                        for _, u, _, _ in tested) if tested else 0.0
            by_covariate = z_any > C.TARONE_Z_THRESHOLD
            over_time = C.temporal_dispersion(series)
            if by_covariate or over_time is not None:
                idx.execute("UPDATE facts SET dispersion_flag=1 WHERE id=?",
                            (fact["id"],))
                if by_covariate:
                    # Show the strongest clustering no guard explains, not the
                    # alphabetically first tested key (Δ10: derived keys sort
                    # early and would otherwise displace the informative one).
                    strongest = max(
                        tested,
                        key=lambda t: C.tarone_z(list(t[1].values())) or 0.0)
                    residual, ruled_out = strongest[1], [t[0] for t in tested]
                else:
                    # The residual partition IS the time blocks: "it is unstable
                    # across these stretches and nothing you logged says why."
                    residual = {f"t{i}": g
                                for i, g in enumerate(over_time[2])}
                    ruled_out = [t[0] for t in tested]
                _open_question(idx, fact["id"], residual, None,
                               ruled_out=ruled_out)

        # §4.6 breadth over confirming observations — RECORDED context only (Δ10):
        # breadth measures the agent's logging diversity; synthesized frames
        # (hour, prev, interactions) must never inflate it.
        rec_keys = sorted({k for ctx in base_ctx for k in ctx})
        confirming = {k: [base_ctx[i][k]
                          for i, (_, out) in enumerate(obs)
                          if out and k in base_ctx[i]]
                      for k in rec_keys}
        report = C.breadth_report(confirming)
        idx.execute("UPDATE facts SET breadth_class=? WHERE id=?",
                    (report["breadth_class"], fact["id"]))
    return proposals


#: Δ11: the prospective audit speaks only past this many scored post-admission
#: observations — the same floor a guard partition needs on each side at entry.
MIN_AUDIT_OBS = 2 * C.MIN_SUPPORT_PER_PARTITION


def prospective_guard_score(idx, body: dict[str, Any],
                            admitted_seq: int) -> Optional[tuple[int, int]]:
    """(hits, misses) of an admitted guard on observations AFTER its admission.

    The entry holdout was scored once, on data that existed at proposal time;
    this is the guard's PROSPECTIVE record — predictions risked on observations
    that did not exist when it was admitted (Δ11). Scoring is identical to the
    entry holdout: predict the outcome (or, for a categorical guard, whether
    the conditioned value occurs) exactly when the conditioning key carries the
    guard's value. Context is augmented with the Δ10 derived frames, so a guard
    on `derived:*` is auditable like any other. Deterministic given the ledger;
    the DECISION over the score belongs to the trusted side (§3.4), not here.

    Returns None when fewer than MIN_AUDIT_OBS post-admission observations are
    scorable — too little rent due to judge.
    """
    fid = body.get("target_fact")
    key = body.get("conditioning_key")
    guards = (body.get("body") or {}).get("guards") or []
    if not fid or not key or not guards:
        return None
    best = guards[0].get("value")
    cval = body.get("conditioned_value")           # set for categorical guards
    rows = idx.query(
        "SELECT o.event_seq, o.outcome, o.value, e.ts FROM observations o "
        "JOIN events e ON e.seq = o.event_seq WHERE o.fact_id=? "
        + ("AND o.value IS NOT NULL " if cval is not None else "")
        + "ORDER BY o.event_seq", (fid,))
    if not rows:
        return None
    ctx_rows = idx.query(
        "SELECT oc.event_seq, oc.key, oc.value FROM obs_context oc "
        "JOIN observations o ON o.event_seq = oc.event_seq "
        "WHERE o.fact_id=?", (fid,))
    by_seq: dict[int, dict[str, str]] = {}
    for r in ctx_rows:
        by_seq.setdefault(int(r["event_seq"]), {})[r["key"]] = r["value"]
    base_ctx = [by_seq.get(int(r["event_seq"]), {}) for r in rows]
    # prev derives from the FULL history (the first post-admission observation's
    # prev is the last pre-admission one), then the score slices post-admission.
    if cval is not None:
        prevs: list = [None] + [rows[i - 1]["value"] for i in range(1, len(rows))]
    else:
        prevs = [None] + ["T" if bool(rows[i - 1]["outcome"]) else "F"
                          for i in range(1, len(rows))]
    aug = C.augment_derived(base_ctx, [int(r["ts"]) for r in rows], prevs)
    hits = misses = 0
    for i, r in enumerate(rows):
        if int(r["event_seq"]) <= admitted_seq or key not in aug[i]:
            continue
        predicted = aug[i][key] == best
        actual = (r["value"] == cval) if cval is not None else bool(r["outcome"])
        if predicted == actual:
            hits += 1
        else:
            misses += 1
    if hits + misses < MIN_AUDIT_OBS:
        return None
    return hits, misses


def _prev_absorbs_time(obs) -> bool:
    """Does conditioning on derived:prev leave both subseries time-stable (Δ10)?

    `obs` is the fact's augmented (ctx, outcome) list in event order. For each
    prev value the conditioned subseries (still in time order) is checked for
    residual temporal dispersion and for a leftover one-way step. Pure function
    of the fact's own observations — resweep purity and replay hold.
    """
    for val in ("T", "F"):
        sub = [out for ctx, out in obs if ctx.get(C.DERIVED_PREV) == val]
        if C.temporal_dispersion(sub) is not None:
            return False
        if C.changepoint_test(sub) is not None:
            return False
    return True


def _sweep_categorical_fact(idx, fact) -> list[tuple[str, dict[str, Any]]]:
    """Per-value ONE-VS-REST sweep for a categorical fact (C4, design §7 — LOCKED).

    For each observed value ``v`` the categorical observation series is projected
    to the binary series ``[value == v]`` and the EXISTING covariate machinery —
    ``partition_by_key → tarone_z → tarone_pvalue → BH → MDL → held-out → guard``
    — runs UNCHANGED on each projection. So "value==captcha is overdispersed on
    region, and captcha concentrates in region=eu" surfaces as a guard on
    ``region=eu`` proposed through the same gate flow.

    Benjamini–Hochberg corrects across the FULL K-values × keys comparison set
    together (design §7's multiple-comparison note), so the per-value multiplicity
    does not inflate the false-guard rate. Like the binary sweep this is a PURE
    function of the fact's OWN observations (the resweep purity contract), so the
    categorical verdict replays / checkpoints bit-for-bit (I3/I8).

    Deferred to the joint-multinomial successor (design §7): a single G-test of
    independence over the value × context table (no K-way tax) and per-value
    changepoint→supersede on the time axis. v1 ships one-vs-rest guards + breadth.
    """
    proposals: list[tuple[str, dict[str, Any]]] = []
    rows = idx.query(
        "SELECT o.event_seq, o.value, e.ts FROM observations o "
        "JOIN events e ON e.seq = o.event_seq "
        "WHERE o.fact_id=? AND o.value IS NOT NULL ORDER BY o.event_seq",
        (fact["id"],))
    if len(rows) < MIN_OBS:
        return proposals
    ctx_rows = idx.query(
        "SELECT oc.event_seq, oc.key, oc.value FROM obs_context oc "
        "JOIN observations o ON o.event_seq = oc.event_seq "
        "WHERE o.fact_id=?", (fact["id"],))
    by_seq: dict[int, dict[str, str]] = {}
    for r in ctx_rows:
        by_seq.setdefault(int(r["event_seq"]), {})[r["key"]] = r["value"]
    # Δ10: same derived-frame augmentation as the binary sweep; `derived:prev`
    # carries the fact's PREVIOUS categorical value. Recorded context is kept
    # separately for breadth.
    base_ctx = [by_seq.get(int(r["event_seq"]), {}) for r in rows]
    prevs: list = [None] + [rows[i - 1]["value"] for i in range(1, len(rows))]
    aug = C.augment_derived(base_ctx, [int(r["ts"]) for r in rows], prevs)
    cat_obs = [(aug[i], r["value"]) for i, r in enumerate(rows)]
    keys = sorted({k for ctx, _ in cat_obs for k in ctx})
    values = sorted({v for _, v in cat_obs})

    # Same held-out discovery/validation split as the binary sweep (M8): keyed on
    # the immutable event_seq, so it is shared across every per-value projection.
    seqs = [int(r["event_seq"]) for r in rows]
    disc = discovery_mask(seqs)
    disc_obs = [o for o, d in zip(cat_obs, disc) if d]
    val_obs = [o for o, d in zip(cat_obs, disc) if not d]

    # Tarone per (value, key) on the one-vs-rest projection; BH across ALL of them.
    tested: list[tuple[str, str, dict[str, C.Group], float, bool]] = []
    for value in values:
        proj = [(ctx, ov == value) for ctx, ov in disc_obs]
        for key in keys:
            groups = C.partition_by_key(proj, key)
            usable = {cv: g for cv, g in groups.items()
                      if g.n >= C.MIN_SUPPORT_PER_PARTITION}
            if len(usable) < 2:
                continue
            z = C.tarone_z(list(usable.values()))
            if z is None:
                continue
            pvalue = C.tarone_pvalue(list(usable.values()))
            if pvalue is None:
                continue
            guardable = len(groups) <= max(
                2, len(disc_obs) // (2 * C.MIN_SUPPORT_PER_PARTITION))
            tested.append((value, key, usable, pvalue, guardable))
    keep = C.benjamini_hochberg([t[3] for t in tested]) if tested else []
    winner = None
    # Δ10: recorded keys outrank derived ones, exactly as in the binary sweep.
    ranked = sorted(zip(keep, tested),
                    key=lambda ft: (C.is_derived(ft[1][1]), ft[1][0], ft[1][1]))
    for flag, (value, key, usable, _, guardable) in ranked:
        if not (flag and guardable):
            continue
        mdl = _mdl_fields(usable)
        if mdl["dl_guard"] + mdl["dl_residual_given_guard"] < mdl["dl_residual"]:
            winner = (value, key, usable, mdl)
            break

    if winner is not None:
        value, key, usable, mdl = winner
        # The context value where value==`value` concentrates (its highest rate).
        best = max(usable, key=lambda cv: usable[cv].k / max(1, usable[cv].n))
        # Guard + direction were chosen on discovery; the gate weighs hits/misses
        # scored on the disjoint validation quarter that no statistic touched (M8):
        # predict value==`value` iff key==best.
        hits = misses = 0
        for ctx, ov in val_obs:
            if key not in ctx:
                continue
            predicted = ctx[key] == best
            actual = ov == value
            if predicted == actual:
                hits += 1
            else:
                misses += 1
        proposal = {
            "head": {"pred": fact["pred"], "args": ["?x"]},
            "body": {"literals": [],
                     "guards": [{"var": f"?{key}", "op": "==", "value": best}]},
            "support": {"left": min(g.n for g in usable.values()),
                        "right": max(g.n for g in usable.values())},
            "holdout": {"hits": hits, "misses": misses},
            "mdl": mdl, "specificity": 1, "conditioning_key": key,
            "conditioned_value": value, "target_fact": fact["id"],
        }
        if C.is_do(key):
            proposal["regime_dependent"] = True        # Δ13, as in the binary path
        proposals.append(("guard", proposal))
        idx.execute("UPDATE facts SET dispersion_flag=1 WHERE id=?",
                    (fact["id"],))
        _open_question(idx, fact["id"], usable, key)

    # §7 breadth for categorical: over ALL observations' recorded context (every
    # categorical observation is informative — there is no single "confirming"
    # outcome to filter on, unlike the binary path). Local to this branch, so
    # binary breadth over confirming observations is untouched. RECORDED keys
    # only (Δ10): synthesized frames never inflate breadth.
    rec_keys = sorted({k for ctx in base_ctx for k in ctx})
    per_key = {k: [ctx[k] for ctx in base_ctx if k in ctx] for k in rec_keys}
    report = C.breadth_report(per_key)
    idx.execute("UPDATE facts SET breadth_class=? WHERE id=?",
                (report["breadth_class"], fact["id"]))
    return proposals


def _open_question(idx, fact_id: str, groups: dict[str, C.Group],
                   explained_key, ruled_out=()) -> None:
    from ..core.hashing import canon_json
    z = C.tarone_z(list(groups.values()))
    if explained_key and C.is_do(explained_key):
        # Δ13: an intervention key does not merely condition — the coupling was
        # regime-dependent, and pre-intervention observation cannot price the
        # post-intervention world.
        suggestion = (f"regime dependence on '{explained_key}': the association "
                      "holds within one intervention regime and does not "
                      "transfer across do(...)")
    elif explained_key:
        suggestion = f"guard proposed on '{explained_key}'"
    elif ruled_out:
        # structure detected, but every recorded key was tested and none
        # cleared the guard bar — the missing argument was never captured
        suggestion = ("variance clusters beyond the recorded keys "
                      f"({', '.join(sorted(set(ruled_out)))}) — log wider: "
                      "capture ambient state you currently have no reason "
                      "to care about, then re-observe")
    else:
        suggestion = C.suggested_measurement([])
    idx.execute(
        "INSERT OR REPLACE INTO open_questions(id, kind, target_kind, target_id, "
        "residual_partition, dispersion_stat, ruled_out_json, "
        "suggested_measurement, status, explained_by_guard_id) "
        "VALUES(?,'dispersion','fact',?,?,?,?,?,?,NULL)",
        (f"q:dispersion:{fact_id}", fact_id,
         canon_json({v: [g.n, g.k] for v, g in sorted(groups.items())}),
         float(z or 0.0), canon_json(sorted(set(ruled_out))), suggestion,
         "explained" if explained_key else "open"))

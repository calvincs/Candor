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


def _mdl_fields(groups: dict[str, C.Group]) -> dict[str, float]:
    pooled = [C.Group(sum(g.n for g in groups.values()),
                      sum(g.k for g in groups.values()))]
    split = list(groups.values())
    bits = 2.0 + math.log2(max(2, len(groups)))          # guard description cost
    return C.mdl_gain(pooled, split, guard_bits=bits)


def sweep(idx) -> list[tuple[str, dict[str, Any]]]:
    """Return (candidate_kind, body) proposals; also set flags/questions."""
    proposals: list[tuple[str, dict[str, Any]]] = []
    # The sweep is a memoryless detector over the raw observation log: a
    # persisting pattern is re-proposed every sweep (gate application is
    # idempotent). Filtering repaired facts here would make detection depend
    # on repair history and break replay-equivalence of the sweep.
    facts = idx.query(
        "SELECT f.id, f.pred, f.args_json, f.stmt_type FROM facts f "
        "ORDER BY f.id")
    for fact in facts:
        rows = idx.query(
            "SELECT o.event_seq, o.outcome, e.ts FROM observations o "
            "JOIN events e ON e.seq = o.event_seq WHERE o.fact_id=? "
            "ORDER BY o.event_seq", (fact["id"],))
        if len(rows) < MIN_OBS:
            continue
        series = [bool(r["outcome"]) for r in rows]
        ctx_rows = idx.query(
            "SELECT oc.event_seq, oc.key, oc.value FROM obs_context oc "
            "JOIN observations o ON o.event_seq = oc.event_seq "
            "WHERE o.fact_id=?", (fact["id"],))
        by_seq: dict[int, dict[str, str]] = {}
        for r in ctx_rows:
            by_seq.setdefault(int(r["event_seq"]), {})[r["key"]] = r["value"]
        obs = [(by_seq.get(int(r["event_seq"]), {}), bool(r["outcome"]))
               for r in rows]
        keys = sorted({k for ctx, _ in obs for k in ctx})

        # covariate search: Tarone per key, BH across the keys tested (§4.5).
        # A key whose cardinality exceeds n/(2*min_support) cannot yield a
        # guard — an m-ary split at that granularity is a lookup table, not a
        # condition — but it still licenses DETECTION (flag + open question).
        tested: list[tuple[str, dict[str, C.Group], float, bool]] = []
        for key in keys:
            groups = C.partition_by_key(obs, key)
            usable = {v: g for v, g in groups.items()
                      if g.n >= C.MIN_SUPPORT_PER_PARTITION}
            if len(usable) < 2:
                continue
            z = C.tarone_z(list(usable.values()))
            if z is None:
                continue
            pvalue = 0.5 * math.erfc(z / math.sqrt(2)) if z > 0 else 1.0
            guardable = len(groups) <= max(
                2, len(obs) // (2 * C.MIN_SUPPORT_PER_PARTITION))
            tested.append((key, usable, pvalue, guardable))
        keep = C.benjamini_hochberg([t[2] for t in tested]) if tested else []
        winner = None
        for flag, (key, usable, _, guardable) in zip(keep, tested):
            if not (flag and guardable):
                continue
            mdl = _mdl_fields(usable)
            if mdl["dl_guard"] + mdl["dl_residual_given_guard"] < mdl["dl_residual"]:
                winner = (key, usable, mdl)
                break

        # §4.4 routing: a regime change is ONE-WAY. Locate the shift at the
        # argmax of cumulative deviation, test it exactly against the search
        # that found it, then ask whether either side changes AGAIN — if so the
        # series oscillates, which is dispersion wearing a changepoint costume,
        # and the repair is a condition or a question, never a supersede.
        detected = C.changepoint_test(series)
        changepoint = detected[0] if detected else None
        changepoint_p = detected[1] if detected else None
        recurrent = C.is_recurrent(series) if detected else False

        if winner is not None:
            key, usable, mdl = winner
            best = max(usable, key=lambda v: usable[v].k / max(1, usable[v].n))
            # §3.4 step 5: validate on held-out observations, not the discovery
            # set — even-indexed obs discover, odd-indexed validate direction.
            hits = misses = 0
            for i, (ctx, out) in enumerate(obs):
                if i % 2 == 0 or key not in ctx:
                    continue
                predicted = ctx[key] == best
                if predicted == out:
                    hits += 1
                else:
                    misses += 1
            proposals.append(("guard", {
                "head": {"pred": fact["pred"], "args": ["?x"]},
                "body": {"literals": [],
                         "guards": [{"var": f"?{key}", "op": "==", "value": best}]},
                "support": {"left": min(g.n for g in usable.values()),
                            "right": max(g.n for g in usable.values())},
                "holdout": {"hits": hits, "misses": misses},
                "mdl": mdl, "specificity": 1, "conditioning_key": key,
                "target_fact": fact["id"],
            }))
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
                    residual, ruled_out = tested[0][1], [t[0] for t in tested]
                else:
                    # The residual partition IS the time blocks: "it is unstable
                    # across these stretches and nothing you logged says why."
                    residual = {f"t{i}": g
                                for i, g in enumerate(over_time[2])}
                    ruled_out = [t[0] for t in tested]
                _open_question(idx, fact["id"], residual, None,
                               ruled_out=ruled_out)

        # §4.6 breadth over confirming observations
        confirming = {k: [ctx[k] for ctx, out in obs if out and k in ctx]
                      for k in keys}
        report = C.breadth_report(confirming)
        idx.execute("UPDATE facts SET breadth_class=? WHERE id=?",
                    (report["breadth_class"], fact["id"]))
    return proposals


def _open_question(idx, fact_id: str, groups: dict[str, C.Group],
                   explained_key, ruled_out=()) -> None:
    from ..core.hashing import canon_json
    z = C.tarone_z(list(groups.values()))
    if explained_key:
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

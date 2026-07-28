"""Replay Pernix's lived history through CANDOR and ask what it learned.

Each extracted event feeds two facts: the specific statement (pred, [arg]) and
the aggregate (pred, ["*"]) whose ctx carries the arg — so per-target streams
get changepoint/dispersion treatment while cross-target structure (method,
protection, kind) can surface as guards on the aggregate. The arg key on the
aggregate is high-cardinality and therefore detection-only under the D20 rail.

Output: what the curiosity engine found — guards (with held-out record),
regime changes located to calendar dates, open questions with suggested
measurements, breadth classes — plus the raw audit trail to check any of it
against the memories it came from.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from candor.system import CandorSystem                    # noqa: E402

OBS = Path("data/bench/pernix_obs.jsonl")
ROOT = Path("data/bench/candor_realworld")
REPORT = Path("data/bench/realworld_report.json")


def day(epoch: int) -> str:
    return datetime.fromtimestamp(epoch, tz=timezone.utc).strftime("%Y-%m-%d")


def main() -> None:
    events = [json.loads(line) for line in OBS.read_text().splitlines() if line]
    print(f"replaying {len(events)} events "
          f"({day(events[0]['epoch'])} .. {day(events[-1]['epoch'])})")
    import shutil
    shutil.rmtree(ROOT, ignore_errors=True)
    system = CandorSystem(ROOT)
    system.set_actor_quota("human:calvin", cand_per_epoch=10_000)
    system.set_actor_quota("agent:pernix", obs_per_epoch=100_000)
    system.set_actor_quota("agent:curiosity", cand_per_epoch=10_000)

    preds = sorted({e["pred"] for e in events})
    for pred in preds:
        for args in ([f"*"],):
            system.assert_({"pred": pred, "args": args, "stmt_type": "frequency"},
                           source="pernix:ingest", actor="human:calvin")
    specific = sorted({(e["pred"], e["arg"]) for e in events})
    counts = Counter((e["pred"], e["arg"]) for e in events)
    for pred, arg in specific:
        if counts[(pred, arg)] >= 8:          # only targets with real history
            system.assert_({"pred": pred, "args": [arg],
                            "stmt_type": "frequency"},
                           source="pernix:ingest", actor="human:calvin")
    system.run_gate()

    for n, ev in enumerate(events, 1):
        agg_ctx = dict(ev["ctx"], target=ev["arg"])
        system.observe({"pred": ev["pred"], "args": ["*"]}, ev["outcome"],
                       agg_ctx, actor="agent:pernix")
        if counts[(ev["pred"], ev["arg"])] >= 8:
            system.observe({"pred": ev["pred"], "args": [ev["arg"]]},
                           ev["outcome"], ev["ctx"], actor="agent:pernix")
        if n % 500 == 0:
            print(f"  ingested {n}/{len(events)}", flush=True)

    runs = system.run_gate()
    guards = [r for r in runs if r["candidate_kind"] == "guard"]
    supersedes = [r for r in runs if r["candidate_kind"] == "supersede_valid_time"]

    guard_rows = []
    for row in system.index.query(
            "SELECT body_json FROM candidates WHERE kind='guard' "
            "AND status='admitted'"):
        body = json.loads(row["body_json"])
        guard_rows.append({
            "pred": body["head"]["pred"], "key": body.get("conditioning_key"),
            "guard": body["body"]["guards"], "holdout": body.get("holdout"),
            "target_fact": body.get("target_fact")})

    cp_rows = []
    seq2epoch = {}
    for i, ev in enumerate(events):
        seq2epoch[i] = ev["epoch"]
    for row in system.index.query(
            "SELECT body_json FROM candidates WHERE kind='supersede_valid_time' "
            "AND status='admitted'"):
        body = json.loads(row["body_json"])
        fact = system.index.one("SELECT pred, args_json FROM facts WHERE id=?",
                                (body["fact_id"],))
        # locate the changepoint's calendar date via that fact's own obs stream
        obs_rows = system.index.query(
            "SELECT o.event_seq, e.ts FROM observations o JOIN events e "
            "ON e.seq = o.event_seq WHERE o.fact_id=? ORDER BY o.event_seq",
            (body["fact_id"],))
        idx = min(int(body.get("changepoint_index", 0)), len(obs_rows) - 1)
        cp_rows.append({
            "fact": f"{fact['pred']}({json.loads(fact['args_json'])[0]})",
            "changepoint_obs_index": idx, "n_obs": len(obs_rows)})

    questions = system.questions()
    breadth = {r["id"]: r["breadth_class"] for r in system.index.query(
        "SELECT id, breadth_class FROM facts WHERE breadth_class IS NOT NULL")}
    flagged = [dict(r) for r in system.index.query(
        "SELECT f.pred, f.args_json, f.breadth_class FROM facts f "
        "WHERE f.dispersion_flag=1")]

    report = {
        "events": len(events),
        "span": [day(events[0]["epoch"]), day(events[-1]["epoch"])],
        "facts_seeded": len(specific) + len(preds),
        "guards_admitted": guard_rows,
        "regime_changes": cp_rows,
        "open_questions": questions,
        "dispersion_flagged": flagged,
        "gate_runs_this_sweep": {"guards": len(guards),
                                 "supersedes": len(supersedes)},
    }
    REPORT.write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items()
                      if k not in ("open_questions",)}, indent=1)[:3000])
    print(f"\nopen questions: {len(questions)}  -> {REPORT}")
    system.close()


if __name__ == "__main__":
    main()

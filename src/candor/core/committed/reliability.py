"""Actor reliability scorer (trusted, spec §3.12; v0.3 Δ1 two-coin model).

Two views of the same trusted evidence, both moved ONLY when a claim settles
via a `deterministic_total` oracle:

  actor_reliability  one-coin agree/disagree Beta. Retained for the alea
                     discount (a frequency trial is a reported world-outcome,
                     not a judgement) and for the frozen v0.2 conformance hooks.
  actor_confusion    two-coin: (tp, fn, fp, tn) integers per (actor, frame).
                     A vote's evidence is its log-likelihood ratio, which lets
                     an asymmetric observer be exactly what it is — an
                     always-yes actor's TRUE vote carries LR ~ 1 (no
                     information) while its FALSE vote would be decisive.
                     One scalar cannot represent that; the 6.8 post-mortem is
                     the demonstration (bench/FINDINGS_6_8.md F2).

Every real-valued number (E[rel], sens, fpr, LR) is a read-time composition;
storage holds integers and the two legacy Beta reals only (I11).
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..index import Index

# Prior for an actor that has never been scored. Deliberately close to 1: the
# discount is a penalty for demonstrated unreliability, not a tax on newcomers.
REL_PRIOR_A = 19.0
REL_PRIOR_B = 1.0

# v0.3 Δ1: read-time confusion priors, same stance as Beta(19, 1) — newcomers
# are assumed informative (E[sens]=0.95, E[fpr]=0.05, prior mass 10). With zero
# settlements a vote composes at LR ≈ 19, never at LR = 1: evidence must be
# able to speak before it has been scored.
SENS_PRIOR = (9.5, 0.5)
FPR_PRIOR = (0.5, 9.5)

# v0.3 Δ2: observations sharing a context_sig compose sub-additively — their
# group's total log-LR is divided by m^GAMMA. 0 would mean independent, 1 fully
# redundant. Read-time constant; if ever fitted, it is fitted on settled train
# claims only and joins the calibration artifact (I8).
GAMMA = 0.5

FRAMES = ("internal", "external")

# The frame under which raw fact counts are composed. Observations are about
# the world; claims carry their own frame (spec §2, claims.frame).
FACT_FRAME = "external"


def expected(idx: "Index", actor: str, frame: str = FACT_FRAME) -> float:
    row = idx.one(
        "SELECT rel_a, rel_b FROM actor_reliability WHERE actor=? AND frame=?",
        (actor, frame))
    a = float(row["rel_a"]) if row else REL_PRIOR_A
    b = float(row["rel_b"]) if row else REL_PRIOR_B
    return a / (a + b)


def set_reliability(idx: "Index", actor: str, frame: str, a: float, b: float) -> None:
    idx.execute(
        "INSERT INTO actor_reliability(actor, frame, rel_a, rel_b) VALUES(?,?,?,?) "
        "ON CONFLICT(actor, frame) DO UPDATE SET rel_a=excluded.rel_a, "
        "rel_b=excluded.rel_b", (actor, frame, float(a), float(b)))
    idx.commit()


def confusion(idx: "Index", actor: str,
              frame: str = FACT_FRAME) -> tuple[int, int, int, int]:
    row = idx.one(
        "SELECT tp, fn, fp, tn FROM actor_confusion WHERE actor=? AND frame=?",
        (actor, frame))
    if row is None:
        return (0, 0, 0, 0)
    return (int(row["tp"]), int(row["fn"]), int(row["fp"]), int(row["tn"]))


def record_confusion(idx: "Index", actor: str, frame: str, vote: bool,
                     outcome: bool) -> None:
    """One integer increment into the two-coin table. Trusted path only."""
    cell = ("tp" if vote else "fn") if outcome else ("fp" if vote else "tn")
    idx.execute(
        "INSERT OR IGNORE INTO actor_confusion(actor, frame, tp, fn, fp, tn) "
        "VALUES(?,?,0,0,0,0)", (actor, frame))
    idx.execute(
        f"UPDATE actor_confusion SET {cell} = {cell} + 1 "
        "WHERE actor=? AND frame=?", (actor, frame))


def rates(conf: tuple[int, int, int, int]) -> tuple[float, float]:
    """(sens, fpr) posterior means under the read-time priors. Reals here only."""
    tp, fn, fp, tn = conf
    sens = (SENS_PRIOR[0] + tp) / (SENS_PRIOR[0] + SENS_PRIOR[1] + tp + fn)
    fpr = (FPR_PRIOR[0] + fp) / (FPR_PRIOR[0] + FPR_PRIOR[1] + fp + tn)
    return sens, fpr


def log_lr(sens: float, fpr: float, vote: bool) -> float:
    """Evidence carried by one vote, given the actor's operating point."""
    sens = min(1.0 - 1e-9, max(1e-9, sens))
    fpr = min(1.0 - 1e-9, max(1e-9, fpr))
    if vote:
        return math.log(sens / fpr)
    return math.log((1.0 - sens) / (1.0 - fpr))


def temper(log_lr_value: float, weight: float) -> float:
    """Scale one vote's evidence by an operator-set trust weight.

    The lever behind `set_reliability`. A weight of 1.0 leaves the vote exactly
    as its confusion ledger earned it; 0.0 silences it (LR = 1, no information);
    0.8 means "count this source's evidence at 80% strength". Same device the
    Δ2 correlation discount already uses — evidence is tempered in log-odds
    space, never by editing a stored count (I11).

    Applies to any log-LR whatever its provenance, so the Δ1 binary path and
    the Δ6 graded path discount identically.
    """
    return log_lr_value * max(0.0, min(1.0, weight))


def grouped_logodds(votes, params, prior_logodds: float = 0.0,
                    gamma: float = GAMMA) -> float:
    """Δ2 composition: votes grouped by context signature, sub-additive within.

    `votes` is an iterable of (actor, vote, context_sig); `params` maps actor
    -> (sens, fpr). Observations with no recorded context form singleton
    groups, i.e. compose independently.
    """
    groups: dict[str, float] = {}
    singletons = 0.0
    sizes: dict[str, int] = {}
    for actor, vote, sig in votes:
        sens, fpr = params[actor]
        contribution = log_lr(sens, fpr, bool(vote))
        if sig is None:
            singletons += contribution
        else:
            groups[sig] = groups.get(sig, 0.0) + contribution
            sizes[sig] = sizes.get(sig, 0) + 1
    total = prior_logodds + singletons
    for sig, subtotal in groups.items():
        total += subtotal / (sizes[sig] ** gamma)
    return total


RESPONSE_ALPHA = 0.5          # Dirichlet smoothing over the response ledger
RESPONSE_MIN_SCORED = 10      # below this, fall back to the binary Δ1 LR
N_GRADES = 4                  # 0 ungraded, 1 weak, 2 firm, 3 strong


def grade_of(confidence) -> int:
    """Δ6 binning at the API boundary. None -> 0 (legacy binary)."""
    if confidence is None:
        return 0
    strength = max(float(confidence), 1.0 - float(confidence))
    if strength < 0.75:
        return 1
    return 2 if strength < 0.9 else 3


def record_response(idx: "Index", actor: str, frame: str, vote: bool,
                    grade: int, outcome: bool) -> None:
    idx.execute(
        "INSERT OR IGNORE INTO actor_response(actor, frame, vote, grade, "
        "n_true, n_false) VALUES(?,?,?,?,0,0)",
        (actor, frame, 1 if vote else 0, int(grade)))
    col = "n_true" if outcome else "n_false"
    idx.execute(
        f"UPDATE actor_response SET {col} = {col} + 1 "
        "WHERE actor=? AND frame=? AND vote=? AND grade=?",
        (actor, frame, 1 if vote else 0, int(grade)))


def response_log_lr(idx: "Index", actor: str, vote: bool, grade: int,
                    frame: str = FACT_FRAME) -> float:
    """Δ6 categorical LR with fallback chain: graded -> binary Δ1 -> prior."""
    rows = idx.query(
        "SELECT vote, grade, n_true, n_false FROM actor_response "
        "WHERE actor=? AND frame=?", (actor, frame))
    total_t = sum(int(r["n_true"]) for r in rows)
    total_f = sum(int(r["n_false"]) for r in rows)
    if total_t + total_f >= RESPONSE_MIN_SCORED:
        k = 2 * N_GRADES
        cell_t = cell_f = 0
        for r in rows:
            if int(r["vote"]) == int(bool(vote)) and int(r["grade"]) == int(grade):
                cell_t, cell_f = int(r["n_true"]), int(r["n_false"])
                break
        p_t = (cell_t + RESPONSE_ALPHA) / (total_t + RESPONSE_ALPHA * k)
        p_f = (cell_f + RESPONSE_ALPHA) / (total_f + RESPONSE_ALPHA * k)
        return math.log(p_t / p_f)
    sens, fpr = rates(confusion(idx, actor, frame))
    return log_lr(sens, fpr, bool(vote))


def grouped_logodds_mixed(votes, lr_of, prior_logodds: float = 0.0,
                          gamma: float = GAMMA) -> float:
    """Δ2 grouping over votes whose per-vote logLR comes from `lr_of(vote)`."""
    groups: dict[str, float] = {}
    sizes: dict[str, int] = {}
    singles = 0.0
    for v in votes:
        contribution = lr_of(v)
        sig = v[-1]
        if sig is None:
            singles += contribution
        else:
            groups[sig] = groups.get(sig, 0.0) + contribution
            sizes[sig] = sizes.get(sig, 0) + 1
    total = prior_logodds + singles
    for sig, subtotal in groups.items():
        total += subtotal / (sizes[sig] ** gamma)
    return total


def score_against_settlement(idx: "Index", fact_id: str, outcome: bool,
                             frame: str = FACT_FRAME) -> list[tuple[str, bool]]:
    """Score every prior observation on `fact_id` against a trusted settlement.

    Moves both trust views from the same event: the one-coin Beta (agree /
    disagree) and the two-coin confusion cell (vote direction × settled
    outcome). Returns the (actor, agreed) pairs, for the resolution payload.
    """
    scored: list[tuple[str, bool]] = []
    rows = idx.query(
        "SELECT actor, outcome, grade FROM observations WHERE fact_id=?", (fact_id,))
    for row in rows:
        vote = bool(row["outcome"])
        record_response(idx, row["actor"], frame, vote, int(row["grade"]),
                        bool(outcome))
        agreed = vote is bool(outcome)
        actor = row["actor"]
        cur = idx.one(
            "SELECT rel_a, rel_b FROM actor_reliability WHERE actor=? AND frame=?",
            (actor, frame))
        a = float(cur["rel_a"]) if cur else REL_PRIOR_A
        b = float(cur["rel_b"]) if cur else REL_PRIOR_B
        if agreed:
            a += 1.0
        else:
            b += 1.0
        set_reliability(idx, actor, frame, a, b)
        record_confusion(idx, actor, frame, vote, bool(outcome))
        scored.append((actor, agreed))
    return scored

"""Count updater — trusted (spec §3.7, invariants I2 / I5 / I11).

This module is the *only* writer to `fact_counts` / `rule_counts`. It consumes
`observation` and `resolution` events exclusively. Nothing on the retrieval
side stream may reach it, and the import-graph audit in §6.2 proves it:
`candor.periphery.retrieval` has no path to this module.

Storage discipline: counts are integers keyed by (fact, actor, channel).
The composed, reliability-discounted view is computed here too, but it is
returned — never written back.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from . import reliability

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..index import Index

# ε-defaults (I5): unobserved gets pseudocount mass, never a hard zero. The
# only hard zero in the system is a '-' pin.
#
# The epistemic prior is NOT shared between statement types, because §4.2 says
# the epi channel means different things for each:
#
#   frequency  epi = p(the reference class is valid as stated). Passing all
#              seven gate steps really is strong evidence of that, and the epi
#              channel is the only thing standing behind it — trials move alea,
#              never epi — so the prior has to carry it.
#   crisp      epi = p(the statement is true). Admission is evidence of
#              well-formedness, not of world-truth: every gate step is a
#              structural or consistency check and none of them observes
#              anything. A shared prior would have the gate quietly asserting
#              that admitted crisp facts are 99% true, which is exactly the
#              conflation §4.2 exists to prevent — and it would take ~100
#              contradicting observations to drag such a fact back to even odds.
EPI_PRIOR_FREQUENCY = (99.0, 1.0)
EPI_PRIOR_CRISP = (1.0, 1.0)      # uniform: truth is what observations decide
ALEA_PRIOR_A = 1.0    # uniform Dirichlet over the outcome rate
ALEA_PRIOR_B = 1.0

CHANNEL_FOR_STMT_TYPE = {"crisp": "epi", "frequency": "alea"}


@dataclass(frozen=True)
class Composed:
    """Read-time composition: Σ_actor raw(actor) × E[rel(actor, frame)]."""
    epi_a: float
    epi_b: float
    alea_n: float
    alea_k: float


def channel_for(stmt_type: str) -> str:
    """§4.2: the statement type decides which channel an observation moves."""
    try:
        return CHANNEL_FOR_STMT_TYPE[stmt_type]
    except KeyError:  # pragma: no cover - guarded upstream by the registry
        raise ValueError(f"unknown stmt_type {stmt_type!r}")


def ensure_row(idx: "Index", fact_id: str, actor: str, channel: str) -> None:
    """Create the zero row that makes an admitted fact's audit trail addressable."""
    idx.execute(
        "INSERT OR IGNORE INTO fact_counts(fact_id, actor, channel, n, k) "
        "VALUES(?,?,?,0,0)", (fact_id, actor, channel))


def apply_observation(idx: "Index", fact_id: str, actor: str, channel: str,
                      outcome: bool) -> None:
    """Integer increment. Counts, never deltas; frozen targets are a no-op."""
    row = idx.one("SELECT numeric FROM facts WHERE id=?", (fact_id,))
    if row is None or row["numeric"] == "frozen":
        return
    ensure_row(idx, fact_id, actor, channel)
    idx.execute(
        "UPDATE fact_counts SET n = n + 1, k = k + ? "
        "WHERE fact_id=? AND actor=? AND channel=?",
        (1 if outcome else 0, fact_id, actor, channel))


def apply_category_observation(idx: "Index", fact_id: str, actor: str,
                               value: str) -> None:
    """Integer increment of a per-value tally (categorical C1, §1.4).

    Exact analog of `apply_observation`: respects numeric='frozen' as a no-op,
    INSERT OR IGNORE a zero row then UPDATE n=n+1. The open vocabulary is
    expressed by a brand-new value simply becoming a new (fact,actor,value) row.
    Integer increment, never a delta — so fold order is irrelevant to the stored
    counts (I3), same guarantee fact_counts already relies on.
    """
    if value is None:
        return
    row = idx.one("SELECT numeric FROM facts WHERE id=?", (fact_id,))
    if row is None or row["numeric"] == "frozen":
        return
    value = str(value)
    idx.execute(
        "INSERT OR IGNORE INTO fact_category_counts(fact_id, actor, value, n) "
        "VALUES(?,?,?,0)", (fact_id, actor, value))
    idx.execute(
        "UPDATE fact_category_counts SET n = n + 1 "
        "WHERE fact_id=? AND actor=? AND value=?", (fact_id, actor, value))


def apply_rule_observation(idx: "Index", rule_id: str, actor: str,
                           outcome: bool) -> None:
    idx.execute(
        "INSERT OR IGNORE INTO rule_counts(rule_id, actor, n, k) VALUES(?,?,0,0)",
        (rule_id, actor))
    idx.execute(
        "UPDATE rule_counts SET n = n + 1, k = k + ? WHERE rule_id=? AND actor=?",
        (1 if outcome else 0, rule_id, actor))


def raw_counts(idx: "Index", fact_ids: Iterable[str]) -> dict[tuple[str, str], tuple[int, int]]:
    """Raw storage truth, unioned over an alias closure if several ids are given."""
    out: dict[tuple[str, str], tuple[int, int]] = {}
    for fid in fact_ids:
        for row in idx.query(
                "SELECT actor, channel, n, k FROM fact_counts WHERE fact_id=? "
                "ORDER BY actor, channel", (fid,)):
            key = (row["actor"], row["channel"])
            n, k = out.get(key, (0, 0))
            out[key] = (n + int(row["n"]), k + int(row["k"]))
    return out


def compose(idx: "Index", fact_ids: Iterable[str]) -> Composed:
    """Read-time composition. Reals appear here and only here (I11)."""
    epi_a = epi_b = alea_n = alea_k = 0.0
    cache: dict[str, float] = {}
    for fid in fact_ids:
        for row in idx.query(
                "SELECT actor, channel, n, k FROM fact_counts WHERE fact_id=?", (fid,)):
            actor = row["actor"]
            if actor not in cache:
                cache[actor] = reliability.expected(idx, actor)
            rel = cache[actor]
            n, k = int(row["n"]), int(row["k"])
            if row["channel"] == "epi":
                epi_a += rel * k
                epi_b += rel * (n - k)
            else:
                alea_n += rel * n
                alea_k += rel * k
    return Composed(epi_a, epi_b, alea_n, alea_k)


def epi_prior(stmt_type: str) -> tuple[float, float]:
    return EPI_PRIOR_CRISP if stmt_type == "crisp" else EPI_PRIOR_FREQUENCY


def posterior_params(composed: Composed, stmt_type: str = "frequency"
                     ) -> tuple[tuple[float, float], tuple[float, float]]:
    """(epi Beta, alea Beta) with ε-priors applied. Read time only."""
    prior_a, prior_b = epi_prior(stmt_type)
    epi = (prior_a + composed.epi_a, prior_b + composed.epi_b)
    alea = (ALEA_PRIOR_A + composed.alea_k,
            ALEA_PRIOR_B + max(0.0, composed.alea_n - composed.alea_k))
    return epi, alea

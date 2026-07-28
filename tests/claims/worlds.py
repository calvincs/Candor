"""Synthetic worlds with planted ground truth, shaped like things agents meet.

Every generator returns data whose *true* structure is known, so a test can ask
"did the substrate recover what was actually there?" rather than "did it return
something plausible?". Null worlds (no structure at all) are first-class: a
detector that cannot stay quiet on them is worthless no matter its recall.

All randomness is drawn from a caller-supplied `random.Random`, so every world
is reproducible from its seed.
"""

from __future__ import annotations

import random
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Optional, Sequence

from candor.system import CandorSystem

# ── store construction ──────────────────────────────────────────────────────

BIG_QUOTA = 10_000_000


def fresh_store(prefix: str = "candor-claims-", actors: Iterable[str] = ()) -> CandorSystem:
    """A store in a temp dir with quotas raised out of the way of the test."""
    m = CandorSystem(Path(tempfile.mkdtemp(prefix=prefix)))
    for a in set(actors) | {"human:me", "agent:curiosity"}:
        m.set_actor_quota(a, obs_per_epoch=BIG_QUOTA, cand_per_epoch=BIG_QUOTA)
    return m


def admit(m: CandorSystem, pred: str, args: Sequence[str],
          stmt_type: str = "frequency", actor: str = "human:me") -> str:
    """Assert a statement and drive it through the gate; return its fact id."""
    m.assert_({"pred": pred, "args": list(args), "stmt_type": stmt_type},
              source="suite", actor=actor)
    m.run_gate()
    return m.fact_id_for({"pred": pred, "args": list(args)})


# ── outcome streams: what a tool's success log looks like over time ─────────

@dataclass(frozen=True)
class Stream:
    """A planted outcome series. `changepoint` is None for stationary worlds."""
    outcomes: list[bool]
    changepoint: Optional[int]
    label: str
    contexts: list[dict] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.outcomes)

    def ctx_at(self, i: int) -> dict:
        return self.contexts[i] if self.contexts else {}


def step(n: int, cp: int, p_before: float, p_after: float,
         rng: random.Random, label: str = "step") -> Stream:
    """A tool that broke (or got fixed) on a specific day. One-way change."""
    return Stream([rng.random() < (p_before if i < cp else p_after)
                   for i in range(n)], cp, label)


def stationary(n: int, p: float, rng: random.Random) -> Stream:
    """NULL WORLD: nothing ever changed. Any changepoint found here is invented."""
    return Stream([rng.random() < p for i in range(n)], None, f"flat p={p}")


def oscillating(n: int, period: int, p_hi: float, p_lo: float,
                rng: random.Random) -> Stream:
    """NULL WORLD for changepoints: it keeps flapping, so no single date is it.

    This is dispersion wearing a changepoint costume — a load shedder, a cache
    that warms and cools, a rate limiter. The honest answer is 'unstable',
    never 'it broke on Tuesday'.
    """
    return Stream([rng.random() < (p_hi if (i // period) % 2 else p_lo)
                   for i in range(n)], None, "oscillating")


def drift(n: int, p_start: float, p_end: float, rng: random.Random) -> Stream:
    """A slow slide with no single break — model rot, growing corpus, decay."""
    return Stream([rng.random() < (p_start + (p_end - p_start) * i / (n - 1))
                   for i in range(n)], None, "gradual drift")


def burst(n: int, start: int, length: int, p_base: float, p_burst: float,
          rng: random.Random) -> Stream:
    """An incident: it broke, then RECOVERED. Not a regime change — an outage."""
    return Stream([rng.random() < (p_burst if start <= i < start + length
                                   else p_base) for i in range(n)],
                  None, "transient outage")


# ── covariate worlds: success that depends on something you did or didn't log ──

def covariate(n: int, rng: random.Random, p_hi: float, p_lo: float,
              key: str = "method", values: Sequence[str] = ("crawl4ai", "http"),
              nuisance: int = 5, hidden_period: int = 0) -> Stream:
    """A stream whose success depends on `key` (or, if hidden_period, on an
    UNLOGGED block variable while `key` is recorded but irrelevant).

    `nuisance` pure-noise covariates are always recorded alongside, because a
    real agent logs plenty of context that explains nothing — and a covariate
    search that cannot survive that is a fishing expedition.
    """
    outcomes, contexts = [], []
    for i in range(n):
        v = values[i % len(values)]
        ctx = {key: v}
        for j in range(nuisance):
            ctx[f"noise{j}"] = rng.choice(["a", "b"])
        if hidden_period:
            p = p_hi if (i // hidden_period) % 2 else p_lo
        else:
            p = p_hi if v == values[0] else p_lo
        outcomes.append(rng.random() < p)
        contexts.append(ctx)
    return Stream(outcomes, None, "covariate", contexts)


def feed(m: CandorSystem, fact_stmt: dict, s: Stream, actor: str = "tool:probe",
         ctx_fn: Optional[Callable[[int], dict]] = None) -> list[int]:
    """Play a stream into the store as attributed observations."""
    seqs = []
    for i, ok in enumerate(s.outcomes):
        ctx = ctx_fn(i) if ctx_fn else s.ctx_at(i)
        seqs.append(m.observe(fact_stmt, ok, ctx, actor=actor))
    return seqs


# ── judge panels: multi-source verification with known operating points ─────

@dataclass(frozen=True)
class Judge:
    name: str
    sens: float          # P(says yes | true)
    fpr: float           # P(says yes | false)

    def vote(self, truth: bool, rng: random.Random) -> bool:
        return rng.random() < (self.sens if truth else self.fpr)


#: The panel that motivated the two-coin model. Each judge is a real failure
#: mode: the sycophant agrees with everything, the specialist almost never
#: fires but is nearly always right when it does, the alarmist cries wolf.
STRONG_PANEL = (
    Judge("agent:careful", 0.95, 0.05),
    Judge("agent:noisy", 0.65, 0.35),
    Judge("agent:sycophant", 1.00, 1.00),
    Judge("agent:specialist", 0.50, 0.02),
    Judge("agent:alarmist", 0.99, 0.60),
)

#: Deliberately weak so posteriors spread across [0,1] instead of piling up at
#: the ends — you cannot measure calibration on a panel that is always sure.
WEAK_PANEL = (
    Judge("agent:w1", 0.75, 0.25),
    Judge("agent:w2", 0.70, 0.30),
    Judge("agent:w3", 0.80, 0.20),
)


def correlated_panel(base: Judge, n_clones: int) -> tuple[Judge, ...]:
    """n judges reading the SAME upstream evidence — the double-counting trap."""
    return tuple(Judge(f"{base.name}-clone{i}", base.sens, base.fpr)
                 for i in range(n_clones))

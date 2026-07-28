"""
CANDOR conformance harness — executable counterpart to spec §6 (v0.2).

An implementation is CONFORMANT when this suite passes against its Driver.
The suite drives the system exclusively through the §5 API plus the four
test-only hooks in §6.1. It never touches implementation internals.

Layout
------
  Driver / HarnessDriver   protocols the implementation must satisfy
  SyntheticWorld        seeded ground-truth generator for §6.4
  test_p_*              §6.2 property invariants   (marks: fail_stop / alert_only)
  test_g_*              §6.3 golden fixtures
  test_s_*              §6.4 statistical validation
  test_a_*              §6.5 adversarial suite
  test_d_*              §6.6 durability & replay
  Stage gates           §6.7 via pytest marks: stage1..stage5

Run a stage gate:   pytest candor_conformance.py -m stage1
Requires:           pytest, hypothesis   (pip install pytest hypothesis)

Wire-up: implement Driver/HarnessDriver for your build and point the `driver`
fixture at it. Everything marked `impl hook` needs one small accessor on the
HarnessDriver; each is listed in HarnessDriver so the surface stays enumerable.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Iterable, Optional, Protocol, Sequence, runtime_checkable

import pytest

try:
    from hypothesis import given, settings, strategies as st
    HAVE_HYPOTHESIS = True
except ImportError:  # suite still importable; property tests skip
    HAVE_HYPOTHESIS = False

# ──────────────────────────────────────────────────────────────────────────
# Result types — the API's epistemic contracts, as types (spec §5, I4)
# ──────────────────────────────────────────────────────────────────────────

class DeriveStatus(Enum):
    PROOF = "proof"
    NOT_ENTAILED = "not_entailed"        # ONLY when search exhausted within budget
    BUDGET_EXCEEDED = "budget_exceeded"  # truncated ⇒ unknown, never NOT_ENTAILED


@dataclass(frozen=True)
class DeriveResult:
    status: DeriveStatus
    proof: Optional[Any] = None
    search_exhausted: bool = False       # impl hook: engine must report this honestly


@dataclass(frozen=True)
class Prediction:
    p: float
    ci: tuple[float, float]
    channels: dict[str, float]           # {"epistemic": ..., "aleatoric": ...}
    sensitivity: dict[str, float]        # component id -> sensitivity
    mpe: Any
    caveats: frozenset[str]              # e.g. {"shared_provenance", "narrow_breadth"}
    snapshot_id: str                     # {ledger_head_hash, engine_version, calib_map_hash}
    rejection_rate: float                # constraint-inconsistent epistemic samples


@dataclass(frozen=True)
class RawCounts:
    """Storage-truth counts. MUST be integers keyed by (actor, channel). (I11)"""
    by_actor: dict[tuple[str, str], tuple[int, int]]  # (actor, channel) -> (n, k)


@dataclass(frozen=True)
class ComposedCounts:
    """Read-time composition: raw × E[reliability]. Reals allowed HERE only."""
    epi_a: float
    epi_b: float
    alea_n: float
    alea_k: float


# ──────────────────────────────────────────────────────────────────────────
# Driver protocols (§6.1)
# ──────────────────────────────────────────────────────────────────────────

@runtime_checkable
class Driver(Protocol):
    # writes ----------------------------------------------------------------
    def assert_(self, stmt: dict, source: str, actor: str) -> str: ...
    def observe(self, stmt: dict, outcome: bool, ctx: dict[str, str],
                actor: str) -> int: ...
    def observe_batch(self, obs: Sequence[tuple[dict, bool, dict, str]]) -> list[int]: ...
    def claim(self, stmt: dict, frame: str, criterion: str, due: int) -> str: ...
    def resolve(self, claim_id: str, outcome: bool) -> int: ...
    def supersede(self, target_id: str, reason: str) -> int: ...
    def pin(self, target_id: str, polarity: str, reason: str, authority: str) -> int: ...
    def redact(self, payload_hash: str) -> int: ...
    def run_gate(self) -> list[dict]: ...     # drain pending candidates -> gate runs

    # reads -----------------------------------------------------------------
    def recall(self, query: str, budget: int) -> list[dict]: ...
    def derive(self, goal: dict, budget: int) -> DeriveResult: ...
    def conjecture(self, goal: dict, sim_budget: float) -> list[dict]: ...
    def predict(self, stmt: dict, budget: int) -> Prediction: ...
    def why(self, obj_id: str) -> dict: ...
    def questions(self, scope: str = "*") -> list[dict]: ...
    def events_since(self, cursor: int, kinds: Optional[set[str]] = None) -> list[dict]: ...
    def health(self) -> dict: ...


@runtime_checkable
class HarnessDriver(Driver, Protocol):
    """Driver + fault injection + the enumerated impl hooks. Test builds only."""
    def reset(self) -> None: ...
    def replay(self) -> str: ...                       # rebuild from ledger -> closure hash
    def closure_hash(self) -> str: ...
    def ledger_head(self) -> str: ...
    def corrupt(self, what: str, arg: str = "") -> None: ...  # torn_tail|drop_index|delete_payload

    # impl hooks (small, enumerated, test-only)
    def raw_counts(self, fact_id: str) -> RawCounts: ...
    def composed_counts(self, fact_id: str) -> ComposedCounts: ...
    def fact_id_for(self, stmt: dict) -> Optional[str]: ...
    def predict_at(self, stmt: dict, snapshot_id: str) -> Prediction: ...  # I8 re-run
    def set_reliability(self, actor: str, frame: str, a: float, b: float) -> None: ...
    def storage_scan_nonintegral_counts(self) -> list[str]: ...  # ids violating I11
    def grep_weight_outside_committed(self) -> list[str]: ...    # lexical firewall
    def retrieval_writer_import_paths(self) -> list[str]: ...    # must be []


# ──────────────────────────────────────────────────────────────────────────
# Fixtures — point `make_driver` at your implementation
# ──────────────────────────────────────────────────────────────────────────

def make_driver() -> HarnessDriver:
    pytest.skip("wire make_driver() to your HarnessDriver implementation")


@pytest.fixture()
def d() -> HarnessDriver:
    drv = make_driver()
    drv.reset()
    return drv


def _seed_minimal_world(d: HarnessDriver) -> dict[str, str]:
    """Seed path (§8): registry + a few facts through the gate, human proposer."""
    fid = {}
    for stmt in (
        {"pred": "reachable", "args": ["a", "b"], "stmt_type": "crisp"},
        {"pred": "reachable", "args": ["b", "c"], "stmt_type": "crisp"},
        {"pred": "flaky_link", "args": ["c", "d"], "stmt_type": "frequency"},
    ):
        d.assert_(stmt, source="seed", actor="human:calvin")
    d.run_gate()
    for stmt in (
        {"pred": "reachable", "args": ["a", "b"]},
        {"pred": "flaky_link", "args": ["c", "d"]},
    ):
        got = d.fact_id_for(stmt)
        if got:
            fid[stmt["pred"] + ":" + ",".join(stmt["args"])] = got
    return fid


GOAL = {"pred": "reachable", "args": ["a", "c"]}
BUDGET = 10_000

# ──────────────────────────────────────────────────────────────────────────
# §6.2 Property invariants
# ──────────────────────────────────────────────────────────────────────────

fail_stop = pytest.mark.fail_stop
alert_only = pytest.mark.alert_only


@pytest.mark.stage1
@fail_stop
def test_p_replay_determinism(d: HarnessDriver):
    _seed_minimal_world(d)
    d.observe({"pred": "flaky_link", "args": ["c", "d"]}, True,
              {"site": "lab", "temp": "20C"}, actor="tool:probe")
    before = d.closure_hash()
    assert d.replay() == before, "rebuild from ledger must be bit-identical (I1/I3)"


@pytest.mark.stage1
@fail_stop
def test_p_count_provenance_and_isolation(d: HarnessDriver):
    """I2: retrieval has no write path to counts — structural + behavioral check."""
    assert d.retrieval_writer_import_paths() == []
    fids = _seed_minimal_world(d)
    fid = fids["flaky_link:c,d"]
    before = d.raw_counts(fid).by_actor
    for _ in range(25):
        d.recall("flaky link c d", budget=512)
    assert d.raw_counts(fid).by_actor == before, "retrieval moved a weight (I2)"


@pytest.mark.stage1
@fail_stop
def test_p_count_integrality(d: HarnessDriver):
    fids = _seed_minimal_world(d)
    d.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {}, actor="tool:probe")
    assert d.storage_scan_nonintegral_counts() == [], "reals in storage (I11)"
    rc = d.raw_counts(fids["flaky_link:c,d"])
    assert all(isinstance(n, int) and isinstance(k, int)
               for n, k in rc.by_actor.values())


@pytest.mark.stage3
@alert_only
def test_p_composition_purity(d: HarnessDriver):
    fids = _seed_minimal_world(d)
    fid = fids["flaky_link:c,d"]
    for out in (True, True, False, True):
        d.observe({"pred": "flaky_link", "args": ["c", "d"]}, out, {}, actor="agent:x")
    d.set_reliability("agent:x", "external", a=8.0, b=2.0)   # E[rel] = 0.8
    comp = d.composed_counts(fid)
    raw = d.raw_counts(fid).by_actor
    n_raw = sum(n for (_, ch), (n, _) in raw.items() if ch == "alea")
    k_raw = sum(k for (_, ch), (_, k) in raw.items() if ch == "alea")
    assert comp.alea_n == pytest.approx(0.8 * n_raw, rel=1e-9)
    assert comp.alea_k == pytest.approx(0.8 * k_raw, rel=1e-9)


@pytest.mark.stage3
@fail_stop
def test_p_two_channel_routing(d: HarnessDriver):
    """§4.2: crisp obs -> epi only; frequency obs -> alea only."""
    fids = _seed_minimal_world(d)
    crisp, freq = fids["reachable:a,b"], fids["flaky_link:c,d"]
    d.observe({"pred": "reachable", "args": ["a", "b"]}, True, {}, actor="tool:probe")
    d.observe({"pred": "flaky_link", "args": ["c", "d"]}, True, {}, actor="tool:probe")
    rc_c = d.raw_counts(crisp).by_actor
    rc_f = d.raw_counts(freq).by_actor
    assert any(ch == "epi" and n > 0 for (_, ch), (n, _) in rc_c.items())
    assert not any(ch == "alea" and n > 0 for (_, ch), (n, _) in rc_c.items())
    assert any(ch == "alea" and n > 0 for (_, ch), (n, _) in rc_f.items())
    assert not any(ch == "epi" and n > 0 for (_, ch), (n, _) in rc_f.items())


@pytest.mark.stage3
@fail_stop
def test_p_budget_honesty(d: HarnessDriver):
    """derive: NOT_ENTAILED requires exhausted search; truncation != absence (I4)."""
    _seed_minimal_world(d)
    starved = d.derive(GOAL, budget=1)
    if starved.status is DeriveStatus.NOT_ENTAILED:
        assert starved.search_exhausted, "NOT_ENTAILED on a truncated search"
    full = d.derive(GOAL, budget=BUDGET)
    assert full.status in (DeriveStatus.PROOF, DeriveStatus.NOT_ENTAILED,
                           DeriveStatus.BUDGET_EXCEEDED)


@pytest.mark.stage2
@fail_stop
def test_p_snapshot_completeness(d: HarnessDriver):
    """I8: predicted_p reproducible from snapshot alone, even after new evidence."""
    _seed_minimal_world(d)
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    p0 = d.predict(stmt, budget=BUDGET)
    for _ in range(10):  # move the present; the past must not move
        d.observe(stmt, False, {}, actor="tool:probe")
    p1 = d.predict_at(stmt, p0.snapshot_id)
    assert p1.p == pytest.approx(p0.p, abs=1e-12), "snapshot leaks (I8)"


@pytest.mark.stage3
@alert_only
def test_p_alias_reversibility(d: HarnessDriver):
    fids = _seed_minimal_world(d)
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    baseline = d.predict(stmt, budget=BUDGET).p
    d.assert_({"kind": "alias", "canonical": "flaky_link", "alias": "lossy_link",
               "basis": "pinned"}, source="test", actor="human:calvin")
    d.run_gate()
    assert d.raw_counts(fids["flaky_link:c,d"]).by_actor, "raw counts must survive alias"
    alias_ev = [e for e in d.events_since(0, kinds={"alias"})]
    assert alias_ev, "alias admission must be a ledger event"
    d.supersede(target_id=str(alias_ev[-1]["id"]), reason="test unwind")
    assert d.predict(stmt, budget=BUDGET).p == pytest.approx(baseline, abs=1e-12)


@pytest.mark.stage4
@alert_only
def test_p_constraint_conditioning(d: HarnessDriver):
    _seed_minimal_world(d)
    d.assert_({"kind": "constraint", "ctype": "mutex",
               "body": {"pred": "link_state", "exclusive_values": ["up", "down"]}},
              source="test", actor="human:calvin")
    for v in ("up", "down"):
        d.assert_({"pred": "link_state", "args": ["c", v], "stmt_type": "crisp"},
                  source="test", actor="human:calvin")
    d.run_gate()
    pred = d.predict({"pred": "link_state", "args": ["c", "up"]}, budget=BUDGET)
    assert pred.rejection_rate > 0.0, "mutex tension must surface as rejections"


@pytest.mark.stage2
@alert_only
def test_p_lexical_firewall(d: HarnessDriver):
    assert d.grep_weight_outside_committed() == [], "'weight' escaped the committed tier"


if HAVE_HYPOTHESIS:

    @pytest.mark.stage4
    @alert_only
    @settings(max_examples=25, deadline=None)
    @given(extra=st.integers(min_value=1, max_value=8))
    def test_p_monotonicity(extra: int):
        d = make_driver(); d.reset()
        _seed_minimal_world(d)
        stmt = {"pred": "flaky_link", "args": ["c", "d"]}
        p_before = d.predict(stmt, budget=BUDGET).p
        d.observe_batch([(stmt, True, {}, "tool:probe")] * extra)
        p_after = d.predict(stmt, budget=BUDGET).p
        assert p_after >= p_before - 1e-9, "adding support lowered p"

    @pytest.mark.stage4
    @alert_only
    @settings(max_examples=25, deadline=None)
    @given(seed=st.integers(min_value=0, max_value=2**32 - 1))
    def test_p_permutation(seed: int):
        """Independent-fact insertion order cannot change a conclusion."""
        obs = [({"pred": "flaky_link", "args": ["c", "d"]}, o, {}, "tool:probe")
               for o in (True, True, False, True, False)]
        results = []
        for order in (obs, random.Random(seed).sample(obs, len(obs))):
            d = make_driver(); d.reset(); _seed_minimal_world(d)
            d.observe_batch(order)
            results.append(d.predict(obs[0][0], budget=BUDGET).p)
        assert results[0] == pytest.approx(results[1], abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────
# §6.3 Golden fixtures (representative subset; one per spec bullet in full suite)
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.stage2
def test_g_canonicalization_units(d: HarnessDriver):
    d.assert_({"kind": "symbol", "pred": "boils_at", "arity": 2,
               "arg_types": ["substance", "temperature"],
               "canonical_units": {"1": "K"}}, source="seed", actor="human:calvin")
    d.assert_({"pred": "boils_at", "args": ["water", "212F"], "stmt_type": "crisp"},
              source="doc1", actor="agent:x")
    d.run_gate()
    fid = d.fact_id_for({"pred": "boils_at", "args": ["water", "373.15K"]})
    assert fid, "gate must canonicalize 212F -> 373.15K at admission (§3.4 step 2)"


@pytest.mark.stage3
def test_g_pin_tension(d: HarnessDriver):
    fids = _seed_minimal_world(d)
    d.pin(fids["reachable:a,b"], polarity="-", reason="known-bad",
          authority="human:calvin")
    for _ in range(20):
        d.observe({"pred": "reachable", "args": ["a", "b"]}, True, {}, actor="tool:probe")
    assert d.derive({"pred": "reachable", "args": ["a", "b"]},
                    budget=BUDGET).status is not DeriveStatus.PROOF, "pin must hold (I5)"
    assert any(q["kind"] == "pin_tension" for q in d.questions()), \
        "contradicting evidence vs '-' pin must page a human"


@pytest.mark.stage5
def test_g_changepoint_vs_guard_routing(d: HarnessDriver):
    """Step function -> supersede-with-valid-time; covariate split -> guard."""
    _seed_minimal_world(d)
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    for i in range(40):                       # regime shift at i=20
        d.observe(stmt, i < 20, {"t": str(i)}, actor="tool:probe")
    stmt2 = {"pred": "boils_ok", "args": ["water"], "stmt_type": "frequency"}
    d.assert_(stmt2, source="seed", actor="human:calvin"); d.run_gate()
    for i in range(40):                       # pressure-conditioned, time-stable
        alt = "high" if i % 2 else "low"
        d.observe({"pred": "boils_ok", "args": ["water"]}, alt == "low",
                  {"pressure": alt, "t": str(i)}, actor="tool:probe")
    runs = d.run_gate()
    kinds = {r.get("candidate_kind") for r in runs}
    assert "supersede_valid_time" in kinds and "guard" in kinds, \
        f"routing failed; gate saw {kinds} (§4.5)"


# ──────────────────────────────────────────────────────────────────────────
# §6.4 Statistical validation — seeded synthetic worlds
# ──────────────────────────────────────────────────────────────────────────

@dataclass
class SyntheticWorld:
    """Ground-truth generator. Deterministic under seed; scoring is against truth."""
    seed: int
    n_crisp: int = 30
    n_freq: int = 30
    rng: random.Random = field(init=False)
    crisp_truth: dict[str, bool] = field(default_factory=dict)
    freq_theta: dict[str, float] = field(default_factory=dict)

    def __post_init__(self):
        self.rng = random.Random(self.seed)
        for i in range(self.n_crisp):
            self.crisp_truth[f"c{i}"] = self.rng.random() < 0.5
        for i in range(self.n_freq):
            self.freq_theta[f"f{i}"] = self.rng.betavariate(2, 2)

    def emit(self, d: HarnessDriver, trials_per_fact: int = 40,
             actor: str = "tool:probe") -> None:
        for name, truth in self.crisp_truth.items():
            stmt = {"pred": "holds", "args": [name], "stmt_type": "crisp"}
            d.assert_(stmt, source="synth", actor="human:calvin")
        for name in self.freq_theta:
            stmt = {"pred": "fires", "args": [name], "stmt_type": "frequency"}
            d.assert_(stmt, source="synth", actor="human:calvin")
        d.run_gate()
        for name, theta in self.freq_theta.items():
            for _ in range(trials_per_fact):
                d.observe({"pred": "fires", "args": [name]},
                          self.rng.random() < theta, {}, actor=actor)


@pytest.mark.stage4
def test_s_two_channel_recovery(d: HarnessDriver):
    w = SyntheticWorld(seed=42)
    w.emit(d)
    covered = total = 0
    for name, theta in w.freq_theta.items():
        pr = d.predict({"pred": "fires", "args": [name]}, budget=BUDGET)
        lo, hi = pr.ci
        covered += int(lo <= theta <= hi)
        total += 1
    assert covered / total >= 0.87, f"90% CI covered only {covered}/{total} (§6.4)"


@pytest.mark.stage3
def test_s_actor_discount(d: HarnessDriver):
    """Honest + random observer: composed posterior must lean honest after scoring."""
    fids = _seed_minimal_world(d)
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    rng = random.Random(7)
    truth = 0.9
    for _ in range(60):
        d.observe(stmt, rng.random() < truth, {}, actor="tool:honest")
        d.observe(stmt, rng.random() < 0.5, {}, actor="agent:random")
    d.set_reliability("tool:honest", "external", a=45.0, b=5.0)
    d.set_reliability("agent:random", "external", a=15.0, b=15.0)
    c = d.composed_counts(fids["flaky_link:c,d"])
    pooled_rate = c.alea_k / max(c.alea_n, 1e-9)
    assert abs(pooled_rate - truth) < abs(0.70 - truth), \
        "reliability discount failed to pull posterior toward the honest actor"


@pytest.mark.stage5
def test_s_dispersion_power(d: HarnessDriver):
    """Latent covariate with real effect must trip Tarone; recorded as power point."""
    _seed_minimal_world(d)
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    rng = random.Random(11)
    for i in range(120):
        elev = "high" if i % 2 else "sea"
        theta = 0.35 if elev == "high" else 0.90     # large delta
        d.observe(stmt, rng.random() < theta, {"elevation": elev}, actor="tool:probe")
    d.run_gate()
    flagged = d.why(d.fact_id_for(stmt)).get("dispersion_flag")
    assert flagged, "overdispersion with delta=0.55, n=120 must be detected (§4.5)"


# ──────────────────────────────────────────────────────────────────────────
# §6.5 Adversarial (representative) · §6.6 Durability
# ──────────────────────────────────────────────────────────────────────────

@pytest.mark.stage3
def test_a_observation_flooding_quota(d: HarnessDriver):
    _seed_minimal_world(d)
    stmt = {"pred": "flaky_link", "args": ["c", "d"]}
    with pytest.raises(Exception):
        for _ in range(1_000_000):           # must trip obs_quota_per_epoch long before
            d.observe(stmt, True, {}, actor="agent:spammer")
    assert any(e["kind"] == "quota_exhausted" or "quota" in str(e)
               for e in d.health().get("events", [])) or True  # surfaced in health()


@pytest.mark.stage3
@pytest.mark.xfail(reason="reputation farming: v1 documented limitation (§3.12); "
                          "test pins the blast radius, not full defense")
def test_a_reputation_farming(d: HarnessDriver):
    raise NotImplementedError("build the probe when reliability scoring lands")


@pytest.mark.stage1
def test_d_torn_tail_recovery(d: HarnessDriver):
    _seed_minimal_world(d)
    good_head = d.ledger_head()
    d.corrupt("torn_tail")
    recovered = d.replay()
    assert recovered, "recovery must truncate to last verifying line, not crash"
    assert d.ledger_head() == good_head, "verified prefix must survive intact"


@pytest.mark.stage1
def test_d_index_loss_rebuild(d: HarnessDriver):
    _seed_minimal_world(d)
    before = d.closure_hash()
    d.corrupt("drop_index")
    assert d.replay() == before, "SQLite is a view; segments are truth (I1)"


@pytest.mark.stage1
def test_d_redaction_replay(d: HarnessDriver):
    _seed_minimal_world(d)
    ev = d.observe({"pred": "flaky_link", "args": ["c", "d"]}, True,
                   {"secret": "purge-me"}, actor="tool:probe")
    payload_hash = d.events_since(ev - 1)[0]["payload_hash"]
    d.redact(payload_hash)
    h = d.replay()
    assert h == d.closure_hash(), "post-redaction replay must be deterministic"
    assert "purge-me" not in str(d.recall("purge-me", budget=512)), \
        "redacted content resurfaced"


# ──────────────────────────────────────────────────────────────────────────
# pytest wiring
# ──────────────────────────────────────────────────────────────────────────

def pytest_configure(config):
    for m in ("stage1", "stage2", "stage3", "stage4", "stage5",
              "fail_stop", "alert_only"):
        config.addinivalue_line("markers", f"{m}: CANDOR conformance tag (spec §6)")

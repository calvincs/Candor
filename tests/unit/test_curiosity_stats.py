"""Curiosity-engine statistics (spec §4.5–4.7).

Stage 5 wires these into gate candidates; the statistics themselves are tested
here so the Stage-5 work starts from a known-good numeric base.
"""

from __future__ import annotations

import random

import pytest

from candor.periphery import curiosity as C


def test_homogeneous_groups_are_not_overdispersed():
    rng = random.Random(5)
    groups = [C.Group(20, sum(1 for _ in range(20) if rng.random() < 0.5))
              for _ in range(12)]
    assert not C.overdispersed(groups)


def test_a_latent_binary_covariate_trips_tarone():
    rng = random.Random(5)
    groups = []
    for i in range(12):
        theta = 0.9 if i % 2 else 0.35
        groups.append(C.Group(20, sum(1 for _ in range(20) if rng.random() < theta)))
    assert C.overdispersed(groups)


def test_tarone_is_undefined_rather_than_wrong_on_degenerate_input():
    assert C.tarone_z([C.Group(10, 10), C.Group(10, 10)]) is None
    assert C.tarone_z([C.Group(10, 5)]) is None


def test_partition_by_key_uses_components_not_the_hash():
    obs = [({"elevation": "sea"}, True), ({"elevation": "sea"}, False),
           ({"elevation": "high"}, False), ({"other": "x"}, True)]
    groups = C.partition_by_key(obs, "elevation")
    assert groups == {"sea": C.Group(2, 1), "high": C.Group(1, 0)}


def test_cusum_finds_a_step_function_and_ignores_a_stable_series():
    step = [True] * 20 + [False] * 20
    assert C.cusum_changepoint(step) is not None
    rng = random.Random(2)
    stable = [rng.random() < 0.5 for _ in range(40)]
    assert C.cusum_changepoint(stable) is None


def test_cusum_declines_to_guess_on_a_short_series():
    assert C.cusum_changepoint([True, False, True]) is None


def test_benjamini_hochberg_bounds_false_discovery():
    rng = random.Random(9)
    spurious = [rng.uniform(0.2, 1.0) for _ in range(20)]
    assert not any(C.benjamini_hochberg(spurious))
    assert C.benjamini_hochberg([0.0001] + spurious)[0]


def test_mdl_prefers_a_genuine_split_and_rejects_an_overfit_one():
    residual = [C.Group(40, 20)]
    genuine = [C.Group(20, 19), C.Group(20, 1)]
    gain = C.mdl_gain(residual, genuine, guard_bits=4.0)
    assert gain["dl_guard"] + gain["dl_residual_given_guard"] < gain["dl_residual"]

    noise = [C.Group(20, 10), C.Group(20, 10)]
    flat = C.mdl_gain(residual, noise, guard_bits=32.0)
    assert flat["dl_guard"] + flat["dl_residual_given_guard"] >= flat["dl_residual"]


def test_breadth_is_per_key_entropy_not_signature_distinctness():
    """§4.6: two contexts differing in one irrelevant key are not maximally diverse."""
    narrow = C.breadth_report({"site": ["lab"] * 10, "run": [str(i) for i in range(10)]})
    assert narrow["per_key"]["site"] == 0.0
    assert narrow["per_key"]["run"] == pytest.approx(1.0)
    assert narrow["breadth_class"] == "moderate"

    broad = C.breadth_report({"site": ["lab", "field", "sea", "alp"] * 3,
                              "run": [str(i) for i in range(12)]})
    assert broad["breadth_class"] == "broad"


def test_transferability_is_capped_by_breadth_independently_of_count():
    assert C.transferability(0.99, "narrow") == 0.5
    assert C.transferability(0.99, "broad") == 0.99
    assert C.transferability(0.2, "broad") == 0.2


def test_a_residual_with_no_shared_covariate_says_log_wider():
    assert "log wider" in C.suggested_measurement([])
    assert "elevation" in C.suggested_measurement(["elevation"])

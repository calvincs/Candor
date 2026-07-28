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


# ── exact changepoint routing (§4.4) ────────────────────────────────────────
# The CUSUM alarm is normalised by sqrt(p(1-p)), a Gaussian approximation that
# fails badly on skewed Bernoulli increments. These pin the exact replacement.

def _fisher_reference(k1, n1, k2, n2):
    """Exact-integer hypergeometric sum. Slow and obviously correct — the
    log-space implementation has to match it, not the other way round."""
    from math import comb
    total_n, total_k = n1 + n2, k1 + k2
    observed = comb(n1, k1) * comb(n2, total_k - k1)
    return sum(comb(n1, x) * comb(n2, total_k - x)
               for x in range(max(0, total_k - n2), min(n1, total_k) + 1)
               if comb(n1, x) * comb(n2, total_k - x) <= observed) / comb(total_n, total_k)


@pytest.mark.parametrize("table", [(1, 10, 11, 14), (5, 10, 5, 10),
                                   (20, 20, 0, 20), (54, 60, 6, 60),
                                   (8, 9, 1, 40)])
def test_fisher_exact_matches_an_exact_integer_reference(table):
    assert C.fisher_exact(*table) == pytest.approx(_fisher_reference(*table),
                                                   rel=1e-9)


def test_changepoint_significance_is_flat_across_base_rates():
    """The defect that motivated this: the old inner CUSUM false-alarmed on
    40% of stationary p=0.95 segments and 3% at p=0.5. An exact test must not
    care what the base rate is."""
    for p in (0.05, 0.1, 0.5, 0.9, 0.95):
        rng = random.Random(hash(p) & 0xFFFF)
        fires = sum(C.changepoint_test([rng.random() < p for _ in range(120)])
                    is not None for _ in range(200))
        assert fires / 200 <= 0.05, f"stationary p={p} alarmed {fires}/200"


def test_a_real_step_is_found_and_located():
    rng = random.Random(4)
    errors = []
    for _ in range(100):
        s = [rng.random() < 0.9 for _ in range(60)] + \
            [rng.random() < 0.1 for _ in range(60)]
        found = C.changepoint_test(s)
        assert found is not None, "missed an unmissable 0.9 -> 0.1 break"
        errors.append(abs(found[0] - 59))
    errors.sort()
    assert errors[len(errors) // 2] <= 2, f"median localization {errors}"
    assert errors[int(0.9 * len(errors))] <= 5, "p90 localization degraded"


def test_oscillation_never_becomes_a_regime_change():
    """The routing contract, not the internals: a flapping service must not
    yield a supersede, whether because no split is significant or because the
    split it finds recurs."""
    proposals = 0
    for seed in range(100):
        rng = random.Random(seed)
        s = [rng.random() < (0.85 if (i // 20) % 2 else 0.35) for i in range(240)]
        found = C.changepoint_test(s)
        if found is not None and not C.is_recurrent(s):
            proposals += 1
    assert proposals <= 5, f"{proposals}/100 flapping series read as a break"


def test_a_clean_step_is_not_recurrent():
    rng = random.Random(8)
    clean = 0
    for _ in range(50):
        s = [rng.random() < 0.95 for _ in range(60)] + \
            [rng.random() < 0.05 for _ in range(60)]
        clean += not C.is_recurrent(s)
    assert clean >= 45, f"recurrence veto ate {50 - clean}/50 genuine steps"

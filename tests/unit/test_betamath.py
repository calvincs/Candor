"""Beta numerics: correctness and the monotonicity prediction depends on."""

from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from candor.core.betamath import (beta_mean, beta_quantile_grid, betainc,
                                  betaincinv)


@pytest.mark.parametrize("a,b,x,expect", [
    (1.0, 1.0, 0.25, 0.25),                    # uniform
    (2.0, 1.0, 0.5, 0.25),                     # I_x(2,1) = x^2
    (1.0, 2.0, 0.5, 0.75),                     # I_x(1,2) = 1-(1-x)^2
    (3.0, 1.0, 0.5, 0.125),
])
def test_betainc_matches_closed_forms(a, b, x, expect):
    assert betainc(a, b, x) == pytest.approx(expect, abs=1e-12)


def test_betainc_endpoints():
    assert betainc(2.0, 5.0, 0.0) == 0.0
    assert betainc(2.0, 5.0, 1.0) == 1.0


@settings(max_examples=100, deadline=None)
@given(a=st.floats(0.5, 40.0), b=st.floats(0.5, 40.0), p=st.floats(1e-4, 1 - 1e-4))
def test_inverse_is_a_right_inverse(a, b, p):
    x = betaincinv(a, b, p)
    assert betainc(a, b, x) == pytest.approx(p, abs=1e-9)


def test_quantile_grid_is_ascending_and_centred():
    grid = beta_quantile_grid(3.0, 7.0, 512)
    assert grid == sorted(grid)
    assert sum(grid) / len(grid) == pytest.approx(beta_mean(3.0, 7.0), abs=2e-3)


def test_grid_is_monotone_in_the_success_count():
    """This is why `predict` is monotone in support rather than merely usually so."""
    lo = beta_quantile_grid(1.0 + 3, 1.0 + 5, 64)
    hi = beta_quantile_grid(1.0 + 4, 1.0 + 4, 64)
    assert all(h >= l for h, l in zip(hi, lo))


def test_grid_resolution_brackets_the_nominal_interval():
    grid = beta_quantile_grid(20.0, 5.0, 512)
    lo = grid[int(0.05 * 511)]
    hi = grid[int(0.95 * 511)]
    assert betainc(20.0, 5.0, lo) == pytest.approx(0.05, abs=2e-3)
    assert betainc(20.0, 5.0, hi) == pytest.approx(0.95, abs=2e-3)

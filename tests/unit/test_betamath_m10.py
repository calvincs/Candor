"""M10: betainc must never silently return a non-probability.

The continued fraction is truncated at 300 terms with no convergence check, so
for a + b beyond ~1e6 it returns wrong values and beyond ~6e8 it returns values
outside [0, 1] (even negative). betainc must stay in [0, 1] everywhere and stay
accurate in the normal operating range.
"""

from __future__ import annotations

from fractions import Fraction
from math import comb

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from candor.core.betamath import betainc


def test_betainc_stays_in_range_for_huge_parameters():
    # a=b=1e9, x=0.5 used to return ~-1.18 (out of range). By symmetry the true
    # value is exactly 0.5; the normal approximation lands there.
    v = betainc(1e9, 1e9, 0.5)
    assert 0.0 <= v <= 1.0, f"betainc out of range: {v}"
    assert v == pytest.approx(0.5, abs=1e-6)


def test_betainc_never_leaves_the_unit_interval():
    for a in (1e5, 1e6, 6e8, 1e9, 5e9, 1e12):
        for x in (0.01, 0.1, 0.5, 0.9, 0.99):
            v = betainc(a, a, x)
            assert 0.0 <= v <= 1.0, f"betainc({a},{a},{x}) = {v} outside [0,1]"


def test_betainc_large_params_is_monotone_and_symmetric():
    # Still a CDF at large parameters: increasing in x, and symmetric about 0.5.
    lo = betainc(1e9, 1e9, 0.49)
    mid = betainc(1e9, 1e9, 0.5)
    hi = betainc(1e9, 1e9, 0.51)
    assert lo < mid < hi
    assert mid == pytest.approx(0.5, abs=1e-6)
    assert lo == pytest.approx(1.0 - hi, abs=1e-6)


def _exact_ibeta(a: int, b: int, x: Fraction) -> Fraction:
    """I_x(a,b) as an exact rational for integer a,b via the binomial identity:
    I_x(a,b) = sum_{k=a}^{a+b-1} C(a+b-1,k) x^k (1-x)^{a+b-1-k}."""
    n = a + b - 1
    om = 1 - x
    total = Fraction(0)
    for k in range(a, n + 1):
        total += comb(n, k) * x ** k * om ** (n - k)
    return total


@pytest.mark.parametrize("a,b,x", [
    (5, 7, 0.3), (20, 5, 0.5), (50, 50, 0.4), (1, 100, 0.05),
    (100, 3, 0.9), (2, 2, 0.25),
])
def test_betainc_matches_exact_binomial_identity(a, b, x):
    exact = float(_exact_ibeta(a, b, Fraction(x)))
    assert betainc(float(a), float(b), x) == pytest.approx(exact, abs=1e-11)


def test_betainc_exact_closed_forms_hold_into_the_large_range():
    # I_x(a,1) = x^a and I_x(1,b) = 1-(1-x)^b are exact; check them at a,b=1e4,
    # inside the normal operating range where the continued fraction is used.
    assert betainc(1e4, 1.0, 0.9997) == pytest.approx(0.9997 ** 1e4, rel=1e-9)
    assert betainc(1.0, 1e4, 0.0003) == pytest.approx(
        1.0 - (1.0 - 0.0003) ** 1e4, rel=1e-9)
    # Symmetric point stays exact at a,b=1e4.
    assert betainc(1e4, 1e4, 0.5) == pytest.approx(0.5, abs=1e-9)


@settings(max_examples=60, deadline=None)
@given(a=st.floats(0.5, 5000.0), b=st.floats(0.5, 5000.0),
       x=st.floats(1e-6, 1 - 1e-6))
def test_betainc_is_always_a_probability(a, b, x):
    v = betainc(a, b, x)
    assert 0.0 <= v <= 1.0

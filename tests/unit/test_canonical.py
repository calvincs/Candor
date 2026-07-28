"""Canonicalizer: unit normal forms and fact identity (spec §3.4 step 2, §3.13)."""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from candor.core.canonical import (CanonicalizationError, canonicalize_arg,
                                   canonicalize_args, context_signature, convert,
                                   fact_key, format_number, parse_quantity)


def test_the_bug_the_registry_exists_to_kill():
    """`boils(water, 212F)` vs `boiling_point(H2O, 100C)` must agree on the number."""
    assert canonicalize_arg("212F", "K") == "373.15K"
    assert canonicalize_arg("100C", "K") == "373.15K"
    assert canonicalize_arg("373.15K", "K") == "373.15K"


def test_canonical_form_is_idempotent():
    once = canonicalize_arg("212F", "K")
    assert canonicalize_arg(once, "K") == once


@pytest.mark.parametrize("text,unit,expect", [
    ("1km", "m", "1000m"),
    ("2.54cm", "m", "0.0254m"),
    ("1atm", "Pa", "101325Pa"),
    ("1h", "s", "3600s"),
])
def test_dimension_conversions(text, unit, expect):
    assert canonicalize_arg(text, unit) == expect


def test_non_quantity_under_declared_unit_is_a_gate_error():
    with pytest.raises(CanonicalizationError):
        canonicalize_arg("tepid", "K")


def test_cross_dimension_conversion_refused():
    with pytest.raises(CanonicalizationError):
        convert(1.0, "m", "K")


def test_unitless_args_are_only_stripped():
    assert canonicalize_args([" water ", "b"], None) == ["water", "b"]


def test_fact_key_is_content_addressed_and_order_sensitive():
    assert fact_key("p", ["a", "b"]) == fact_key("p", ["a", "b"])
    assert fact_key("p", ["a", "b"]) != fact_key("p", ["b", "a"])
    assert fact_key("p", ["a"]) != fact_key("q", ["a"])


def test_context_signature_ignores_key_order_only():
    assert context_signature({"a": "1", "b": "2"}) == context_signature({"b": "2", "a": "1"})
    assert context_signature({"a": "1"}) != context_signature({"a": "2"})
    assert context_signature({}) is None


@settings(max_examples=200, deadline=None)
@given(v=st.floats(min_value=-500.0, max_value=5000.0, allow_nan=False,
                   allow_infinity=False))
def test_kelvin_roundtrip_is_stable_through_the_string_form(v):
    text = format_number(v) + "K"
    parsed = parse_quantity(text)
    assert parsed is not None
    assert canonicalize_arg(text, "K") == text

"""Canonicalizer: unit normalization + argument normal forms (spec §3.4 step 2).

Trusted *harness* over untrusted *content*: the conversion table is
deterministic and registry-driven; the proposal being normalized is not.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from .hashing import canon_json, sha256_hex

# dimension -> unit -> (scale, offset) into the dimension's base unit
UNIT_TABLE: dict[str, dict[str, tuple[float, float]]] = {
    "temperature": {
        "K": (1.0, 0.0),
        "C": (1.0, 273.15),
        "F": (5.0 / 9.0, 273.15 - 32.0 * 5.0 / 9.0),
    },
    "length": {
        "m": (1.0, 0.0), "km": (1000.0, 0.0), "cm": (0.01, 0.0),
        "mm": (0.001, 0.0), "ft": (0.3048, 0.0), "in": (0.0254, 0.0),
        "mi": (1609.344, 0.0),
    },
    "mass": {"kg": (1.0, 0.0), "g": (0.001, 0.0), "lb": (0.45359237, 0.0)},
    "pressure": {
        "Pa": (1.0, 0.0), "kPa": (1000.0, 0.0), "bar": (100000.0, 0.0),
        "atm": (101325.0, 0.0), "hPa": (100.0, 0.0),
    },
    "time": {"s": (1.0, 0.0), "ms": (0.001, 0.0), "min": (60.0, 0.0),
             "h": (3600.0, 0.0)},
}

_UNIT_OF: dict[str, str] = {}
for _dim, _units in UNIT_TABLE.items():
    for _u in _units:
        _UNIT_OF[_u] = _dim

_QUANTITY = re.compile(r"^\s*([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)\s*([A-Za-z°]+)\s*$")


class CanonicalizationError(ValueError):
    pass


def format_number(value: float) -> str:
    """Round-trip-stable rendering. 10 significant digits kills FP conversion dust."""
    if value == 0.0:
        value = 0.0            # collapse -0.0; sign of zero is not information
    text = f"{value:.10g}"
    if text.endswith(".0"):
        text = text[:-2]
    return text


def parse_quantity(text: str) -> Optional[tuple[float, str]]:
    m = _QUANTITY.match(text)
    if not m:
        return None
    unit = m.group(2).replace("°", "")
    if unit not in _UNIT_OF:
        return None
    return float(m.group(1)), unit


def convert(value: float, unit: str, target: str) -> float:
    dim = _UNIT_OF.get(unit)
    if dim is None or _UNIT_OF.get(target) != dim:
        raise CanonicalizationError(f"cannot convert {unit!r} to {target!r}")
    scale, offset = UNIT_TABLE[dim][unit]
    base = value * scale + offset
    tscale, toffset = UNIT_TABLE[dim][target]
    return (base - toffset) / tscale


def canonicalize_arg(arg: Any, canonical_unit: Optional[str]) -> Any:
    """Normalize one argument. Non-quantities are stripped strings, untouched."""
    if isinstance(arg, (int, float, bool)) or arg is None:
        return arg
    text = str(arg).strip()
    if canonical_unit:
        parsed = parse_quantity(text)
        if parsed is None:
            raise CanonicalizationError(
                f"argument {arg!r} is not a quantity but the registry declares "
                f"canonical unit {canonical_unit!r}")
        value, unit = parsed
        return format_number(convert(value, unit, canonical_unit)) + canonical_unit
    return text


def canonicalize_args(args: list[Any],
                      canonical_units: Mapping[str, str] | None) -> list[Any]:
    units = canonical_units or {}
    return [canonicalize_arg(a, units.get(str(i))) for i, a in enumerate(args)]


def fact_key(pred: str, args: list[Any]) -> str:
    """Content-addressed fact identity. Stable across replay and process runs."""
    return "fact:" + sha256_hex(canon_json([pred, args]))[:32]


def context_signature(ctx: Mapping[str, str] | None) -> Optional[str]:
    """§4.6: hash of the canonical serialization of the structured obs_context.

    Fast grouping only; covariate search operates on the components, never this.
    """
    if not ctx:
        return None
    return sha256_hex(canon_json(sorted((str(k), str(v)) for k, v in ctx.items())))

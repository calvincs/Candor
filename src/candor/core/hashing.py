"""Canonical serialization and hashing. Trusted (spec §1, §3.1)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

GENESIS = "0" * 64


def canon_json(obj: Any) -> str:
    """Deterministic JSON: sorted keys, no insignificant whitespace."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_hex(data: bytes | str) -> str:
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def hash_obj(obj: Any) -> str:
    """Content address of a payload: hex sha256 of its canonical serialization."""
    return sha256_hex(canon_json(obj))


def stable_u64(text: str) -> int:
    """Deterministic 64-bit integer from text. Used for seed derivation only."""
    return int(sha256_hex(text)[:16], 16)

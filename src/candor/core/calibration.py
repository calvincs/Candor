"""Calibration bucketer and read-time isotonic map (trusted, spec §2, §3.9, I8/I9).

Partitioned by (frame, settlement, predictor_class) — never pooled. Buckets
hold INTEGER tallies only; mean_p and observed_freq are ratios computed on read.

The isotonic map is applied at read time and its hash is part of every model
snapshot. Stored counts are never mutated by calibration.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Optional, Sequence

from .hashing import canon_json, sha256_hex

if TYPE_CHECKING:  # pragma: no cover - typing only
    from .index import Index

N_BUCKETS = 10
MIN_N_FOR_ALERT = 50          # §2: alerting requires n >= min_n per bucket
ENGINE_VERSION = "candor-engine/0.2.0"
DEFAULT_PREDICTOR_CLASS = "wmc-two-loop/v1"


def bucket_of(p: float) -> int:
    return min(N_BUCKETS - 1, max(0, int(p * N_BUCKETS)))


@dataclass(frozen=True)
class IsotonicMap:
    """Monotone piecewise-linear recalibration, x and y ascending."""
    xs: tuple[float, ...] = ()
    ys: tuple[float, ...] = ()

    @property
    def hash(self) -> str:
        return sha256_hex(canon_json([list(self.xs), list(self.ys)]))[:16]

    def apply(self, p: float) -> float:
        if not self.xs:
            return p
        if p <= self.xs[0]:
            return self.ys[0]
        if p >= self.xs[-1]:
            return self.ys[-1]
        for i in range(1, len(self.xs)):
            if p <= self.xs[i]:
                x0, x1 = self.xs[i - 1], self.xs[i]
                y0, y1 = self.ys[i - 1], self.ys[i]
                if x1 == x0:
                    return y1
                return y0 + (y1 - y0) * (p - x0) / (x1 - x0)
        return self.ys[-1]

    def to_json(self) -> str:
        return canon_json({"xs": list(self.xs), "ys": list(self.ys)})

    @staticmethod
    def from_json(text: Optional[str]) -> "IsotonicMap":
        if not text:
            return IsotonicMap()
        obj = json.loads(text)
        return IsotonicMap(tuple(obj.get("xs", ())), tuple(obj.get("ys", ())))


def fit_isotonic(pairs: Sequence[tuple[float, int]]) -> IsotonicMap:
    """Pool-adjacent-violators on (predicted p, observed 0/1) pairs.

    Trained on settled claims only — never on read-path data (§6.4).
    """
    if not pairs:
        return IsotonicMap()
    ordered = sorted(pairs, key=lambda t: t[0])
    # Pool tied x into ONE block BEFORE PAVA: `sorted` is stable, so equal-x
    # points otherwise keep their input order and PAVA merges them differently
    # depending on it — and predict() emits multiples of 1/512, so tied x is the
    # norm. Pooling first makes the fit a function of the input multiset, not the
    # order settled claims happened to be enumerated in (M7).
    pooled: list[list[float]] = []  # [sum_y, count, x] — one entry per distinct x
    for x, y in ordered:
        if pooled and pooled[-1][2] == float(x):
            pooled[-1][0] += float(y)
            pooled[-1][1] += 1.0
        else:
            pooled.append([float(y), 1.0, float(x)])
    blocks: list[list[float]] = []  # [sum_y, count, sum_x]
    for sum_y, count, x in pooled:
        blocks.append([sum_y, count, x * count])
        while len(blocks) >= 2 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            a = blocks.pop()
            blocks.append([a[0] + b[0], a[1] + b[1], a[2] + b[2]])
    xs, ys = [], []
    for sum_y, count, sum_x in blocks:
        xs.append(sum_x / count)
        ys.append(sum_y / count)
    return IsotonicMap(tuple(xs), tuple(ys))


def record(idx: "Index", frame: str, settlement: str, predictor_class: str,
           predicted_p: float, outcome: bool, ts: int) -> None:
    b = bucket_of(predicted_p)
    idx.execute(
        "INSERT OR IGNORE INTO calibration(frame, settlement, predictor_class, "
        "bucket, n, k, p_milli, updated_at) VALUES(?,?,?,?,0,0,0,?)",
        (frame, settlement, predictor_class, b, ts))
    idx.execute(
        "UPDATE calibration SET n = n + 1, k = k + ?, p_milli = p_milli + ?, "
        "updated_at = ? WHERE frame=? AND settlement=? AND predictor_class=? "
        "AND bucket=?",
        (1 if outcome else 0, int(round(predicted_p * 1000)), ts,
         frame, settlement, predictor_class, b))


def report(idx: "Index") -> list[dict[str, Any]]:
    out = []
    for row in idx.query(
            "SELECT * FROM calibration ORDER BY frame, settlement, predictor_class, bucket"):
        n = int(row["n"])
        out.append({
            "frame": row["frame"], "settlement": row["settlement"],
            "predictor_class": row["predictor_class"], "bucket": int(row["bucket"]),
            "n": n,
            "mean_p": (int(row["p_milli"]) / 1000.0 / n) if n else None,
            "observed_freq": (int(row["k"]) / n) if n else None,
            "alertable": n >= MIN_N_FOR_ALERT,
        })
    return out


def brier(pairs: Sequence[tuple[float, int]]) -> Optional[float]:
    if not pairs:
        return None
    return sum((p - y) ** 2 for p, y in pairs) / len(pairs)


def log_loss(pairs: Sequence[tuple[float, int]], eps: float = 1e-12) -> Optional[float]:
    if not pairs:
        return None
    total = 0.0
    for p, y in pairs:
        q = min(1.0 - eps, max(eps, p))
        total += -(math.log(q) if y else math.log(1.0 - q))
    return total / len(pairs)


def surprisal(predicted_p: float, outcome: bool, eps: float = 1e-12) -> float:
    q = min(1.0 - eps, max(eps, predicted_p))
    return -math.log(q if outcome else 1.0 - q)


def snapshot_id(ledger_head: str, calib_map_hash: str,
                engine_version: str = ENGINE_VERSION) -> str:
    """I8: {ledger head hash, engine version, calibration map hash}, decodable."""
    return canon_json({"ledger_head": ledger_head, "engine_version": engine_version,
                       "calib_map_hash": calib_map_hash})


def parse_snapshot(snapshot: str) -> dict[str, str]:
    return json.loads(snapshot)

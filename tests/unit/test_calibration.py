"""Calibration bucketer, isotonic map, snapshots (spec §3.9, I8/I9)."""

from __future__ import annotations

import random

import pytest

from candor.core import calibration as C


def test_buckets_partition_the_unit_interval():
    assert C.bucket_of(0.0) == 0
    assert C.bucket_of(0.999) == C.N_BUCKETS - 1
    assert C.bucket_of(1.0) == C.N_BUCKETS - 1


def test_identity_map_when_unfitted():
    m = C.IsotonicMap()
    assert m.apply(0.37) == 0.37


def test_pava_output_is_monotone():
    rng = random.Random(3)
    pairs = [(rng.random(), rng.randint(0, 1)) for _ in range(400)]
    m = C.fit_isotonic(pairs)
    assert list(m.ys) == sorted(m.ys)
    assert list(m.xs) == sorted(m.xs)


def test_isotonic_fit_improves_brier_on_a_miscalibrated_predictor():
    rng = random.Random(11)
    raw, truth = [], []
    for _ in range(2000):
        true_p = rng.random()
        stated = true_p ** 2          # systematically under-confident
        raw.append(stated)
        truth.append(1 if rng.random() < true_p else 0)
    split = len(raw) // 2
    m = C.fit_isotonic(list(zip(raw[:split], truth[:split])))
    held = list(zip(raw[split:], truth[split:]))
    before = C.brier(held)
    after = C.brier([(m.apply(p), y) for p, y in held])
    assert after < before


def test_map_hash_changes_with_the_map():
    a = C.IsotonicMap()
    b = C.fit_isotonic([(0.1, 0), (0.9, 1)])
    assert a.hash != b.hash


def test_map_round_trips_through_json():
    m = C.fit_isotonic([(0.1, 0), (0.4, 0), (0.6, 1), (0.9, 1)])
    assert C.IsotonicMap.from_json(m.to_json()) == m


def test_snapshot_carries_the_three_things_i8_requires():
    snap = C.snapshot_id("headhash", "maphash")
    parsed = C.parse_snapshot(snap)
    assert set(parsed) == {"ledger_head", "engine_version", "calib_map_hash"}
    assert parsed["ledger_head"] == "headhash"
    assert parsed["calib_map_hash"] == "maphash"


def test_calibration_is_partitioned_never_pooled(sys_):
    idx = sys_.index
    C.record(idx, "internal", "entailed", "p1", 0.9, True, 0)
    C.record(idx, "external", "entailed", "p1", 0.9, False, 0)
    rows = {(r["frame"], r["settlement"], r["predictor_class"], r["bucket"]): r
            for r in C.report(idx)}
    assert len(rows) == 2, "distinct partitions must not share a bucket (I9)"
    assert rows[("internal", "entailed", "p1", 9)]["observed_freq"] == 1.0
    assert rows[("external", "entailed", "p1", 9)]["observed_freq"] == 0.0


def test_alerting_requires_min_n(sys_):
    idx = sys_.index
    for _ in range(C.MIN_N_FOR_ALERT - 1):
        C.record(idx, "external", "entailed", "p1", 0.5, True, 0)
    assert C.report(idx)[0]["alertable"] is False
    C.record(idx, "external", "entailed", "p1", 0.5, True, 0)
    assert C.report(idx)[0]["alertable"] is True


def test_calibration_tallies_stay_integral(sys_):
    C.record(sys_.index, "external", "entailed", "p1", 0.37, True, 0)
    assert sys_.index.nonintegral_counts() == []


def test_surprisal_is_higher_for_the_unexpected_outcome():
    assert C.surprisal(0.99, False) > C.surprisal(0.99, True)
    assert C.log_loss([(0.5, 1)]) == pytest.approx(0.6931471805599453)

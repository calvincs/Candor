"""Ledger: chain integrity, CAS, torn-write recovery, redaction (spec §3.1)."""

from __future__ import annotations

import json

import pytest

from candor.core.hashing import GENESIS
from candor.core.ledger import Ledger, LedgerError


@pytest.fixture()
def ledger(tmp_path):
    lg = Ledger(tmp_path / "ledger")
    lg.open()
    yield lg
    lg.close()


def test_genesis_head_is_zero(ledger):
    assert ledger.head() == GENESIS
    assert ledger.seq() == 0


def test_chain_links_and_verifies(ledger):
    a = ledger.append("assertion", "human:calvin", {"x": 1})
    b = ledger.append("observation", "tool:probe", {"x": 2})
    assert b.prev_hash == a.hash
    assert ledger.head() == b.hash
    assert ledger.verify_chain()


def test_kind_outside_the_check_set_is_refused(ledger):
    with pytest.raises(LedgerError):
        ledger.append("retrieval", "agent:x", {})


def test_cas_deduplicates_identical_payloads(ledger):
    a = ledger.append("observation", "tool:probe", {"same": True})
    b = ledger.append("observation", "tool:probe", {"same": True})
    assert a.payload_hash == b.payload_hash
    assert len(list(ledger.cas_dir.iterdir())) == 1


def test_torn_tail_truncates_to_last_verifying_line(ledger, tmp_path):
    ledger.append("assertion", "human:calvin", {"x": 1})
    good = ledger.append("assertion", "human:calvin", {"x": 2})
    ledger.append_raw_line('{"seq": 99, "ts": 0, "kind": "obser')

    recovered = Ledger(tmp_path / "ledger")
    recovered.open()
    assert recovered.head() == good.hash
    assert recovered.seq() == good.seq
    assert recovered.verify_chain()
    recovered.close()


def test_tampered_line_is_truncated_not_trusted(ledger, tmp_path):
    first = ledger.append("assertion", "human:calvin", {"x": 1})
    ledger.append("assertion", "human:calvin", {"x": 2})
    ledger.close()
    seg = sorted((tmp_path / "ledger" / "segments").glob("*.jsonl"))[-1]
    lines = seg.read_text().splitlines()
    rec = json.loads(lines[1])
    rec["actor"] = "agent:impostor"
    lines[1] = json.dumps(rec)
    seg.write_text("\n".join(lines) + "\n")

    recovered = Ledger(tmp_path / "ledger")
    recovered.open()
    assert recovered.head() == first.hash, "a tampered line must not survive recovery"
    recovered.close()


def test_redaction_leaves_the_chain_verifying(ledger):
    ev = ledger.append("observation", "tool:probe", {"secret": "purge-me"})
    ledger.append("redaction", "human:operator", {"payload_hash": ev.payload_hash})
    assert ledger.delete_payload(ev.payload_hash) is True
    assert ledger.payload(ev.payload_hash) is None
    assert ledger.verify_chain(), "the chain covers commitments, not payloads"


def test_segments_roll_over(tmp_path, monkeypatch):
    import candor.core.ledger as mod
    monkeypatch.setattr(mod, "SEGMENT_LINES", 4)
    lg = Ledger(tmp_path / "ledger")
    lg.open()
    for i in range(9):
        lg.append("observation", "tool:probe", {"i": i})
    lg.close()
    assert len(sorted((tmp_path / "ledger" / "segments").glob("*.jsonl"))) == 3
    again = Ledger(tmp_path / "ledger")
    again.open()
    assert again.seq() == 9
    assert again.verify_chain()
    again.close()

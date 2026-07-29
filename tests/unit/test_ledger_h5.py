"""Phase 2 FIX H5 — torn-newline recovery corruption (spec §3.1).

A segment whose last line lost ONLY its trailing "\\n" (but still parses and
hash-verifies) must be re-terminated on recovery. Otherwise the next append()
merges onto that line, verify_chain() then raises instead of returning False,
and a later reopen silently discards events.
"""

from __future__ import annotations

import pytest

from candor.core.hashing import canon_json
from candor.core.ledger import Ledger, _event_hash


@pytest.fixture()
def ledger(tmp_path):
    lg = Ledger(tmp_path / "ledger")
    lg.open()
    yield lg
    lg.close()


def test_h5_torn_newline_is_renormalized_on_recovery(tmp_path):
    """(i) A last line missing only its "\\n" is re-terminated; a later append
    starts a fresh physical line instead of merging."""
    root = tmp_path / "ledger"
    lg = Ledger(root)
    lg.open()
    lg.append("pin", "human:calvin", {"x": 1})       # DURABLE
    lg.append("admission", "gate", {"x": 2})          # DURABLE
    lg.close()

    seg = sorted((root / "segments").glob("*.jsonl"))[-1]
    raw = seg.read_bytes()
    assert raw.endswith(b"\n")
    # Torn write: the last line lost ONLY its trailing newline. It still parses
    # and hash-verifies, so recovery accepts it.
    seg.write_bytes(raw[:-1])
    assert not seg.read_bytes().endswith(b"\n")

    recovered = Ledger(root)
    recovered.open()                                   # recovery runs here
    # (a) the segment must be re-normalized back onto disk.
    assert seg.read_bytes().endswith(b"\n"), \
        "recovery must re-terminate the last line so the next append is a fresh line"
    assert recovered.seq() == 2

    # append a third event: it must NOT merge onto the previous line.
    recovered.append("resolution", "gate", {"x": 3})
    assert len(seg.read_bytes().splitlines()) == 3, \
        "third append merged onto the un-terminated line"
    assert recovered.verify_chain() is True
    recovered.close()

    # a fresh reopen keeps all three events.
    again = Ledger(root)
    again.open()
    assert again.seq() == 3
    assert again.verify_chain() is True
    again.close()


def test_h5_verify_chain_returns_false_on_garbage_tail(ledger):
    """(ii) verify_chain() returns False on an unparseable tail; it must not
    raise."""
    ledger.append("assertion", "human:calvin", {"x": 1})
    ledger.append("assertion", "human:calvin", {"x": 2})
    ledger.append_raw_line('{"seq": 3, "ts": 0, "kind": "obser')  # torn / garbage
    assert ledger.verify_chain() is False


def test_h5_recovery_truncates_before_wrong_seq_across_segments(tmp_path, monkeypatch):
    """(iii) GAP-1/2: a well-formed, correctly-hashed line whose seq is wrong
    (here seq+2) is truncated by recovery, even in a later segment."""
    import candor.core.ledger as mod
    monkeypatch.setattr(mod, "SEGMENT_LINES", 4)

    root = tmp_path / "ledger"
    lg = Ledger(root)
    lg.open()
    for i in range(6):                                 # rolls into a 2nd segment
        lg.append("observation", "tool:probe", {"i": i})
    good_seq = lg.seq()
    good_head = lg.head()
    lg.close()

    assert len(sorted((root / "segments").glob("*.jsonl"))) == 2

    # Craft a self-consistent line (hash covers its own fields) whose seq is
    # wrong: it should be good_seq+1 but we write good_seq+2.
    bad_seq = good_seq + 2
    row = {
        "seq": bad_seq, "ts": 0, "kind": "observation", "actor": "x",
        "payload_hash": "0" * 64, "source_ref": None, "context_sig": None,
        "prev_hash": good_head,
        "hash": _event_hash(bad_seq, 0, "observation", "x", "0" * 64,
                            None, None, good_head),
    }
    seg2 = sorted((root / "segments").glob("*.jsonl"))[-1]
    with seg2.open("ab") as fh:
        fh.write((canon_json(row) + "\n").encode("utf-8"))

    recovered = Ledger(root)
    recovered.open()
    assert recovered.seq() == good_seq, "recovery must truncate the wrong-seq line"
    assert recovered.head() == good_head
    assert recovered.verify_chain() is True
    recovered.close()

"""redact() must not let its payload_hash escape the CAS directory (M4).

`redact(payload_hash)` flowed the argument unvalidated into
`cas_dir / f"{payload_hash}.json"`, so a value like "../../../victim" unlinked
an arbitrary .json file outside the store. The fix validates the hash against
^[0-9a-f]{64}$ before any filesystem use.
"""

from __future__ import annotations

import os

import pytest

from candor.system import CandorSystem


def _store_with_secret(root):
    m = CandorSystem(root)
    for stmt in ({"pred": "flaky_link", "args": ["c", "d"],
                  "stmt_type": "frequency"},):
        m.assert_(stmt, source="seed", actor="human:calvin")
    m.run_gate()
    return m


def test_redact_cannot_delete_outside_the_store(tmp_path):
    root = tmp_path / "store"
    m = _store_with_secret(root)

    victim = tmp_path / "victim.json"
    victim.write_text('{"secret": "do not delete me"}', encoding="utf-8")

    cas_dir = root / "ledger" / "payloads"
    # "<traversal>.json" appended to cas_dir resolves onto the victim file.
    traversal = os.path.relpath(tmp_path / "victim", cas_dir)
    assert ".." in traversal, "test setup: expected a traversal path"

    try:
        m.redact(traversal)
    except ValueError:
        pass  # refusing is the correct behavior
    m.close()

    assert victim.exists(), \
        "redact() escaped cas_dir and deleted a file outside the store"


def test_redact_rejects_a_non_hex_hash_with_a_clear_error(tmp_path):
    root = tmp_path / "store"
    m = _store_with_secret(root)
    head_before = m.ledger_head()
    with pytest.raises(ValueError):
        m.redact("not-a-real-hash")
    assert m.ledger_head() == head_before, "a rejected redaction still recorded"
    m.close()


def test_real_redaction_still_purges_the_payload(tmp_path):
    root = tmp_path / "store"
    m = _store_with_secret(root)
    ev = m.observe({"pred": "flaky_link", "args": ["c", "d"]}, True,
                   {"secret": "purge-me"}, actor="tool:probe")
    payload_hash = m.events_since(ev - 1)[0]["payload_hash"]
    assert m.ledger.payload(payload_hash) is not None

    m.redact(payload_hash)

    assert m.ledger.payload(payload_hash) is None, "real payload was not purged"
    assert m.ledger.verify_chain()
    assert "purge-me" not in str(m.recall("purge-me", budget=512)), \
        "redacted content resurfaced through recall"
    m.close()

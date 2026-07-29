"""Hardening of the synthesized-verifier sandbox (spec §3.4 step 3).

The verifier `code` in an `assert_` payload is untrusted content that the gate
runs to admit an oracle. Before this hardening the only isolation was a
20-odd-name ``__builtins__`` allow-list, which is escapable with nothing but
literals and attribute access::

    for c in ().__class__.__base__.__subclasses__():
        if c.__name__ == "BuiltinImporter":
            c.load_module("posix").system("<arbitrary shell>")

These tests pin the two layered controls added in the fix:

  * LAYER 1 — an AST + substring validator that rejects the escape family
    (dunder attribute/name access, imports, calls outside the builtin
    allow-list) *before* the code is ever handed to ``exec``.
  * LAYER 2 — POSIX ``setrlimit`` caps on the child process.

plus coverage for the sandbox failure modes the existing suite never exercised.
"""

from __future__ import annotations

import pytest

from candor.core import sandbox


# ── LAYER 1: the escape must never reach exec ───────────────────────────────

def _escape_code(marker) -> str:
    """A verifier body that walks to BuiltinImporter and shells out.

    `posix` is a builtin module, so ``load_module`` succeeds where "os" fails.
    On unhardened code this writes ``marker`` on the host; hardened, it must be
    rejected before execution and the file must never appear.
    """
    return (
        "def check(x):\n"
        "    for c in ().__class__.__base__.__subclasses__():\n"
        "        if c.__name__ == 'BuiltinImporter':\n"
        f"            c.load_module('posix').system('touch {marker}')\n"
        "    return x + 1\n"
    )


def test_subclasses_escape_is_rejected_by_check_vectors(tmp_path):
    """SECURITY: the classic subclasses/BuiltinImporter walk is blocked and
    the side-effecting shell command never runs."""
    marker = tmp_path / "escaped_direct"
    ok, detail = sandbox.check_vectors(_escape_code(marker), "check", [((1,), 2)])
    assert ok is False
    assert detail  # a descriptive reason, not empty
    assert not marker.exists(), "verifier code executed a shell escape (touch)"


def test_subclasses_escape_is_rejected_by_run(tmp_path):
    marker = tmp_path / "escaped_run"
    res = sandbox.run(_escape_code(marker), "check", [(1,)])
    assert res.ok is False
    assert not marker.exists(), "verifier code executed a shell escape (touch)"


def test_escape_rejected_end_to_end_and_no_oracle_admitted(sys_, tmp_path):
    """SECURITY, public API: assert_ + run_gate on the escape yields a step-3
    verifier rejection, admits no oracle, and writes no marker file."""
    marker = tmp_path / "escaped_e2e"
    sys_.assert_(
        {"kind": "verifier", "oracle_id": "verifier:evil", "entry": "check",
         "code": _escape_code(marker), "vectors": [[[1], 2]]},
        source="seed", actor="human:calvin")
    runs = sys_.run_gate()

    verifier_runs = [r for r in runs if r["candidate_kind"] == "verifier"]
    assert verifier_runs, "no verifier candidate reached the gate"
    assert verifier_runs[0]["status"] == "rejected"
    assert verifier_runs[0]["failing_step"] == 3
    assert not marker.exists(), "malicious verifier executed on the host"
    row = sys_.index.one("SELECT id FROM oracles WHERE id=?", ("verifier:evil",))
    assert row is None, "a malicious verifier was admitted as an oracle"


# ── LAYER 1: the AST/substring validator, unit level ────────────────────────

def test_validator_rejects_dunder_attribute_access():
    ok, detail = sandbox.validate_code(
        "def check(x):\n    return x.__class__\n")
    assert ok is False and "forbidden" in detail


def test_validator_rejects_dunder_names():
    for src in ("def check(x):\n    return __import__\n",
                "def check(x):\n    return __builtins__\n"):
        ok, detail = sandbox.validate_code(src)
        assert ok is False and "forbidden" in detail, src


def test_validator_rejects_import_statements():
    for src in ("import os\ndef check(x):\n    return 1\n",
                "def check(x):\n    from sys import argv\n    return 1\n"):
        ok, detail = sandbox.validate_code(src)
        assert ok is False and "import" in detail, src


def test_validator_rejects_getattr_and_other_offlist_calls():
    ok, detail = sandbox.validate_code(
        "def check(x):\n    return getattr(x, 'foo')\n")
    assert ok is False and "getattr" in detail
    # eval/exec/open/type are absent at runtime; the validator blocks them too.
    for bad in ("eval", "exec", "open", "type", "vars", "globals"):
        src = f"def check(x):\n    return {bad}(x)\n"
        assert sandbox.validate_code(src)[0] is False, bad


def test_validator_substring_backstop_catches_format_string_escape():
    """A dunder that lives only inside a string literal evades AST attribute
    inspection; the raw-substring backstop still rejects it."""
    ok, detail = sandbox.validate_code(
        "def check(x):\n    return '{0.__class__}'.format(())\n")
    assert ok is False and "forbidden" in detail


def test_validator_accepts_clean_arithmetic_verifiers():
    for src in ("def check(x):\n    return x + 1\n",
                "def check(xs):\n    return sum(sorted(xs))\n",
                "def helper(n):\n    return n * 2\n"
                "def check(x):\n    return helper(x) + max(1, 2)\n"):
        ok, detail = sandbox.validate_code(src)
        assert ok is True and detail == "", src


# ── sandbox failure modes (coverage gap at sandbox.py:72-91) ────────────────

def test_timeout_is_rejected_with_a_timeout_reason():
    # A busy loop needs no builtins and never returns; the wall clock fires.
    code = "def check(x):\n    while x == x:\n        pass\n"
    ok, detail = sandbox.check_vectors(code, "check", [((1,), 1)], timeout=0.5)
    assert ok is False and "timeout" in detail


def test_verifier_that_raises_is_rejected():
    ok, detail = sandbox.check_vectors(
        "def check(x):\n    return 1 // 0\n", "check", [((1,), 0)])
    assert ok is False and "ZeroDivisionError" in detail


def test_nonzero_exit_is_rejected(monkeypatch):
    class _Proc:
        returncode = 7
        stdout = ""
        stderr = "boom"

    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: _Proc())
    res = sandbox.run("def check(x):\n    return x\n", "check", [(1,)])
    assert res.ok is False and "sandbox exited 7" in res.error


def test_unparseable_stdout_is_rejected(monkeypatch):
    class _Proc:
        returncode = 0
        stdout = "this is not json at all"
        stderr = ""

    monkeypatch.setattr(sandbox.subprocess, "run", lambda *a, **k: _Proc())
    res = sandbox.run("def check(x):\n    return x\n", "check", [(1,)])
    assert res.ok is False and "parseable" in res.error


def test_empty_vectors_is_rejected():
    ok, detail = sandbox.check_vectors(
        "def check(x):\n    return x\n", "check", [])
    assert ok is False and "vector" in detail


# ── regression: legitimate verifiers still admitted ─────────────────────────

def test_valid_arithmetic_verifier_passes_its_vectors():
    ok, detail = sandbox.check_vectors(
        "def check(x):\n    return x + 1\n", "check", [((1,), 2), ((41,), 42)])
    assert ok is True and detail == ""


def test_valid_verifier_admitted_end_to_end(sys_):
    sys_.assert_(
        {"kind": "verifier", "oracle_id": "verifier:inc2", "entry": "check",
         "code": "def check(x):\n    return x + 1\n",
         "vectors": [[[1], 2], [[41], 42]]},
        source="seed", actor="human:calvin")
    runs = sys_.run_gate()
    verifier_runs = [r for r in runs if r["candidate_kind"] == "verifier"]
    assert verifier_runs and verifier_runs[0]["status"] == "admitted"
    row = sys_.index.one("SELECT kind, code_hash FROM oracles WHERE id=?",
                         ("verifier:inc2",))
    assert row is not None and row["kind"] == "deterministic_total"

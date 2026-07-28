"""Sandboxed execution for synthesized verifiers (trusted harness, spec §3.4 step 3).

The synthesized code is untrusted content; the harness around it is trusted.
Hermeticity here means: a separate isolated interpreter (`-I`, so no user site
packages and no inherited environment), an empty working directory, a hard
timeout, and no arguments passed in beyond the declared test vectors. The parent
process never executes the candidate's code.

Oracles are recorded with `code_hash` and `env_hash` so a settlement stays
auditable after the verifier is patched (§2, §6.6).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Sequence

from .hashing import canon_json, sha256_hex

TIMEOUT_SECONDS = 5.0

_RUNNER = r'''
import json, sys
payload = json.loads(sys.stdin.read())
namespace = {"__builtins__": {
    "abs": abs, "all": all, "any": any, "bool": bool, "dict": dict, "divmod": divmod,
    "enumerate": enumerate, "float": float, "int": int, "len": len, "list": list,
    "max": max, "min": min, "pow": pow, "range": range, "round": round,
    "sorted": sorted, "str": str, "sum": sum, "tuple": tuple, "zip": zip,
}}
try:
    exec(payload["code"], namespace)
    fn = namespace[payload["entry"]]
    results = [fn(*case) for case in payload["cases"]]
    print(json.dumps({"ok": True, "results": results}))
except BaseException as exc:
    print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}))
'''


@dataclass(frozen=True)
class SandboxResult:
    ok: bool
    results: list[Any]
    error: str = ""


def env_hash() -> str:
    """Pinned-environment fingerprint recorded on every resolution (§3.8)."""
    return sha256_hex(canon_json([sys.version_info[:3], sys.platform]))[:16]


def code_hash(code: str) -> str:
    return sha256_hex(code)[:16]


def run(code: str, entry: str, cases: Sequence[Sequence[Any]],
        timeout: float = TIMEOUT_SECONDS) -> SandboxResult:
    payload = canon_json({"code": code, "entry": entry,
                          "cases": [list(c) for c in cases]})
    with tempfile.TemporaryDirectory(prefix="candor-sandbox-") as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _RUNNER],
                input=payload, capture_output=True, text=True, timeout=timeout,
                cwd=cwd, env={"PATH": os.defpath})
        except subprocess.TimeoutExpired:
            return SandboxResult(False, [], "sandbox timeout")
    if proc.returncode != 0:
        return SandboxResult(False, [], f"sandbox exited {proc.returncode}: "
                                        f"{proc.stderr.strip()[:200]}")
    try:
        out = json.loads(proc.stdout.strip().splitlines()[-1])
    except (ValueError, IndexError):
        return SandboxResult(False, [], "sandbox produced no parseable result")
    if not out.get("ok"):
        return SandboxResult(False, [], out.get("error", "unknown sandbox failure"))
    return SandboxResult(True, out.get("results", []))


def check_vectors(code: str, entry: str,
                  vectors: Sequence[tuple[Sequence[Any], Any]],
                  timeout: float = TIMEOUT_SECONDS) -> tuple[bool, str]:
    """Run the declared test vectors. A verifier that cannot pass them is not one."""
    if not vectors:
        return False, "verifier candidate declares no test vectors"
    outcome = run(code, entry, [v[0] for v in vectors], timeout)
    if not outcome.ok:
        return False, outcome.error
    for (args, expected), got in zip(vectors, outcome.results):
        if got != expected:
            return False, f"vector {list(args)} returned {got!r}, expected {expected!r}"
    return True, ""

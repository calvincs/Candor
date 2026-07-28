"""Sandboxed execution for synthesized verifiers (trusted harness, spec §3.4 step 3).

The synthesized code is untrusted content; the harness around it is trusted.
Hermeticity here means: a separate isolated interpreter (`-I`, so no user site
packages and no inherited environment), an empty working directory, a hard
timeout, and no arguments passed in beyond the declared test vectors. The parent
process never executes the candidate's code.

The builtin allow-list alone is *not* a jail — it is escapable with only
literals and attribute access (``().__class__.__base__.__subclasses__()`` ->
``BuiltinImporter`` -> ``posix.system``). Isolation is therefore layered
(defence-in-depth, stdlib-only, in-process):
  LAYER 1  ``validate_code`` statically rejects the escape family (dunder
           attribute/name access, imports, off-allow-list calls) before exec.
  LAYER 2  POSIX ``setrlimit`` caps CPU / address space / open files / procs on
           the child, on top of the wall-clock timeout and ``-I -S`` hygiene.

Oracles are recorded with `code_hash` and `env_hash` so a settlement stays
auditable after the verifier is patched (§2, §6.6).
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence

from .hashing import canon_json, sha256_hex

try:  # POSIX-only stdlib; on other platforms LAYER 2 degrades to a no-op.
    import resource
except ImportError:  # pragma: no cover - non-POSIX platforms
    resource = None  # type: ignore[assignment]

TIMEOUT_SECONDS = 5.0

#: Builtin names exposed to verifier `code`. MUST stay identical to the
#: ``__builtins__`` dict built inside ``_RUNNER`` below; LAYER 1 uses it as a
#: call-site allow-list so a name absent at runtime is also rejected before exec.
_ALLOWED_BUILTINS = frozenset({
    "abs", "all", "any", "bool", "dict", "divmod", "enumerate", "float", "int",
    "len", "list", "max", "min", "pow", "range", "round", "sorted", "str", "sum",
    "tuple", "zip",
})

# LAYER 2 — child resource ceilings (POSIX). The verifier has no `open`, so
# NOFILE/NPROC are pure defence-in-depth against a future gap; AS caps a memory
# blowup and CPU/NPROC blunt a busy-loop or fork bomb that slips the wall clock.
_RLIMIT_AS_BYTES = 256 * 1024 * 1024   # 256 MiB address space
_RLIMIT_NOFILE = 64                    # small open-file ceiling
_RLIMIT_NPROC = 64                     # blunt anti-fork-bomb ceiling

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


def validate_code(code: str) -> tuple[bool, str]:
    """LAYER 1: statically reject the escape family *before* exec.

    The 20-odd-name ``__builtins__`` allow-list is escapable with nothing but
    literals and attribute access — ``().__class__.__base__.__subclasses__()``
    reaches ``BuiltinImporter`` and thence ``posix.system``. This is not a
    provable jail; it is a layered control that denies the known constructs the
    whole escape family requires. A verifier over arithmetic test vectors never
    legitimately needs any of them.

    Rejected (returns ``(False, reason)``):
      * any ``import`` / ``from ... import`` statement,
      * attribute access whose name contains ``__`` (``.__class__``,
        ``.__subclasses__``, ``.__globals__``, ``.__mro__``, …),
      * an identifier containing ``__`` (``__import__``, ``__builtins__``, …),
      * a call to a bare name outside :data:`_ALLOWED_BUILTINS` that is not a
        function defined in the same snippet (blocks ``getattr``/``eval``/
        ``exec``/``open``/``type``/… even though they are absent at runtime),
      * as a blunt backstop, any source containing ``__`` at all — this also
        catches format-string escapes such as ``"{0.__class__}".format(())``
        whose payload hides in a string literal and evades AST inspection.
    """
    try:
        tree = ast.parse(code, mode="exec")
    except SyntaxError as exc:
        return False, f"verifier code does not parse: {exc}"

    local_defs = {node.name for node in ast.walk(tree)
                  if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            return False, ("verifier code uses a forbidden construct: "
                           "import statements are not allowed in the sandbox")
        if isinstance(node, ast.Attribute) and "__" in node.attr:
            return False, ("verifier code uses a forbidden construct: "
                           f"dunder attribute access .{node.attr}")
        if isinstance(node, ast.Name) and "__" in node.id:
            return False, ("verifier code uses a forbidden construct: "
                           f"dunder name {node.id!r}")
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            fname = node.func.id
            if fname not in _ALLOWED_BUILTINS and fname not in local_defs:
                return False, ("verifier code uses a forbidden construct: "
                               f"call to disallowed name {fname!r}")

    # Blunt backstop: no legitimate arithmetic verifier needs a dunder token,
    # and this rejects escapes that live entirely inside string literals.
    if "__" in code:
        return False, ("verifier code uses a forbidden construct: "
                       "double-underscore tokens are not allowed")
    return True, ""


def _rlimit_preexec(cpu_seconds: int) -> Callable[[], None]:
    """LAYER 2: build a POSIX ``preexec_fn`` that caps the child's resources."""
    def _apply() -> None:  # pragma: no cover - runs post-fork in the child
        for name, limit in (("RLIMIT_CPU", cpu_seconds),
                            ("RLIMIT_AS", _RLIMIT_AS_BYTES),
                            ("RLIMIT_NOFILE", _RLIMIT_NOFILE),
                            ("RLIMIT_NPROC", _RLIMIT_NPROC)):
            what = getattr(resource, name, None)
            if what is None:
                continue
            try:
                resource.setrlimit(what, (limit, limit))
            except (ValueError, OSError):
                pass
    return _apply


def run(code: str, entry: str, cases: Sequence[Sequence[Any]],
        timeout: float = TIMEOUT_SECONDS) -> SandboxResult:
    ok, reason = validate_code(code)
    if not ok:
        return SandboxResult(False, [], reason)
    payload = canon_json({"code": code, "entry": entry,
                          "cases": [list(c) for c in cases]})
    preexec: Optional[Callable[[], None]] = None
    if (os.name == "posix" and resource is not None
            and hasattr(resource, "setrlimit")):
        # CPU ceiling comfortably above the wall clock so the timeout wins for a
        # sleeping/looping child; the floor keeps a tiny test override sane.
        preexec = _rlimit_preexec(int(timeout) + 3)
    with tempfile.TemporaryDirectory(prefix="candor-sandbox-") as cwd:
        try:
            proc = subprocess.run(
                [sys.executable, "-I", "-S", "-c", _RUNNER],
                input=payload, capture_output=True, text=True, timeout=timeout,
                cwd=cwd, env={"PATH": os.defpath}, preexec_fn=preexec)
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

"""Mechanical audits behind the §6.2 property invariants.

Two of the invariants are statements about the *source tree*, not about a run:

  count provenance  — the retrieval writer must have no import path to the
                      count updater (I2). Verified by walking the static import
                      graph, not by trusting a comment.
  lexical firewall  — nothing named "weight" exists outside the committed tier
                      (§3.2). Verified by grep over the package.

Both return the offending items, so an empty list is the passing answer.
"""

from __future__ import annotations

import ast
import re
from collections import deque
from pathlib import Path
from typing import Iterable, Optional

PACKAGE_ROOT = Path(__file__).resolve().parent
PACKAGE_NAME = "candor"

# Spec §3.2: this is the committed tier. "weight" is allowed to exist here and
# nowhere else in the package.
COMMITTED_TIER = PACKAGE_ROOT / "core" / "committed"

RETRIEVAL_WRITER = f"{PACKAGE_NAME}.periphery.retrieval"
COUNT_UPDATER = f"{PACKAGE_NAME}.core.committed.counts"

_LEXICAL = re.compile(r"weight", re.IGNORECASE)


def module_path(module: str) -> Optional[Path]:
    if module == PACKAGE_NAME:
        candidate = PACKAGE_ROOT / "__init__.py"
        return candidate if candidate.exists() else None
    if not module.startswith(PACKAGE_NAME + "."):
        return None
    rel = module[len(PACKAGE_NAME) + 1:].split(".")
    direct = PACKAGE_ROOT.joinpath(*rel).with_suffix(".py")
    if direct.exists():
        return direct
    package = PACKAGE_ROOT.joinpath(*rel) / "__init__.py"
    return package if package.exists() else None


def direct_imports(module: str) -> list[str]:
    """In-package modules imported by `module`, including under TYPE_CHECKING."""
    path = module_path(module)
    if path is None:
        return []
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    parent = module.rsplit(".", 1)[0] if not path.name == "__init__.py" else module
    out: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] == PACKAGE_NAME:
                    out.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.level:
                base = parent.split(".")
                drop = node.level - 1
                base = base[:len(base) - drop] if drop else base
                prefix = ".".join(base)
                target = f"{prefix}.{node.module}" if node.module else prefix
            elif node.module and node.module.split(".")[0] == PACKAGE_NAME:
                target = node.module
            else:
                continue
            out.append(target)
            for alias in node.names:
                out.append(f"{target}.{alias.name}")
    resolved = []
    for name in out:
        if module_path(name) is not None and name not in resolved:
            resolved.append(name)
    return resolved


def import_paths(source: str, target: str) -> list[str]:
    """Every acyclic import path from `source` to `target`, as arrow strings."""
    found: list[str] = []
    queue: deque[list[str]] = deque([[source]])
    seen: set[str] = set()
    while queue:
        path = queue.popleft()
        node = path[-1]
        if node == target and len(path) > 1:
            found.append(" -> ".join(path))
            continue
        if node in seen:
            continue
        seen.add(node)
        for nxt in direct_imports(node):
            if nxt in path:
                continue
            queue.append(path + [nxt])
    return sorted(found)


def retrieval_writer_import_paths() -> list[str]:
    """I2, structural half: must be empty, forever."""
    return import_paths(RETRIEVAL_WRITER, COUNT_UPDATER)


def python_files(root: Path = PACKAGE_ROOT) -> Iterable[Path]:
    return sorted(p for p in root.glob("**/*.py"))


# The auditor and the test-only driver adapter must name the check in order to
# perform it — the hook's name is fixed by the conformance protocol. They are
# audit machinery, not part of the substrate, and are the ONLY exemptions
# beyond the committed tier itself. See DEVIATIONS.md D4.
AUDIT_SELF = ("audit.py", "harness.py")


def _exempt(path: Path) -> bool:
    resolved = path.resolve()
    try:
        resolved.relative_to(COMMITTED_TIER.resolve())
        return True
    except ValueError:
        pass
    return resolved.parent == PACKAGE_ROOT and resolved.name in AUDIT_SELF


def grep_weight_outside_committed(root: Path = PACKAGE_ROOT,
                                  evidence_dirs: Iterable[Path] = ()) -> list[str]:
    """§6.2 lexical firewall: `grep -rn weight` outside the committed tier.

    Covers the package source *and* the evidence tier, which is where the
    hazard actually lives: `@weight` was renamed `@salience` precisely so a
    retrieval-ranking input could never be mistaken for a committed number (I2).
    """
    hits: list[str] = []
    targets = [p for p in python_files(root) if not _exempt(p)]
    for directory in evidence_dirs:
        directory = Path(directory)
        if directory.exists():
            targets.extend(sorted(directory.glob("**/*.md")))
    for path in targets:
        try:
            text = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError):
            continue
        for lineno, line in enumerate(text.splitlines(), start=1):
            if _LEXICAL.search(line):
                hits.append(f"{path}:{lineno}:{line.strip()}")
    return hits

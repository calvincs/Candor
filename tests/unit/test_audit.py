"""The two source-tree invariants (spec §6.2 count provenance, lexical firewall).

These are statements about the repository, so they are tested against the
repository — including negative controls, so an always-empty answer cannot pass
for a firewall.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from candor import audit


def test_retrieval_writer_cannot_reach_the_count_updater():
    assert audit.retrieval_writer_import_paths() == []


def test_retrieval_module_imports_nothing_from_the_package():
    assert audit.direct_imports(audit.RETRIEVAL_WRITER) == [], \
        "the retrieval side stream must stay stdlib-only (I2)"


def test_the_import_walker_finds_paths_that_do_exist():
    """Negative control: the walker is not vacuously returning []."""
    paths = audit.import_paths("candor.system", audit.COUNT_UPDATER)
    assert paths, "system.py does reach the count updater; the walker must see it"
    assert all(p.startswith("candor.system") for p in paths)


def test_the_import_walker_follows_transitive_edges():
    paths = audit.import_paths("candor.system", "candor.core.betamath")
    assert any("candor.periphery.predict" in p for p in paths), \
        "betamath is reached only through predict; the walk must be transitive"


def test_the_core_never_imports_the_periphery():
    offenders = []
    for path in audit.python_files(audit.PACKAGE_ROOT / "core"):
        rel = path.relative_to(audit.PACKAGE_ROOT).with_suffix("")
        module = "candor." + ".".join(rel.parts).replace(".__init__", "")
        for imported in audit.direct_imports(module):
            if imported.startswith("candor.periphery"):
                offenders.append(f"{module} -> {imported}")
    assert offenders == [], "the trusted core must not depend on untrusted code"


def test_lexical_firewall_over_the_package():
    assert audit.grep_weight_outside_committed() == []


def test_lexical_firewall_catches_an_evidence_tier_leak(tmp_path):
    """Negative control: a `@weight` in the evidence tier must be reported."""
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "note.md").write_text("@salience: 0.5\nfine\n", encoding="utf-8")
    assert audit.grep_weight_outside_committed(evidence_dirs=[evidence]) == []

    (evidence / "leak.md").write_text("@weight: 0.9\n", encoding="utf-8")
    hits = audit.grep_weight_outside_committed(evidence_dirs=[evidence])
    assert len(hits) == 1 and "leak.md" in hits[0]


def test_the_committed_tier_is_where_weights_are_allowed():
    """Negative control the other way: the exemption is real, not decorative."""
    committed = audit.COMMITTED_TIER
    text = "\n".join(p.read_text(encoding="utf-8") for p in committed.glob("*.py"))
    assert "weight" in text.lower()
    assert audit.grep_weight_outside_committed() == []


def test_exemptions_are_exactly_the_audit_surface():
    assert audit.AUDIT_SELF == ("audit.py", "harness.py")
    assert audit._exempt(audit.PACKAGE_ROOT / "audit.py")
    assert audit._exempt(audit.COMMITTED_TIER / "counts.py")
    assert not audit._exempt(audit.PACKAGE_ROOT / "system.py")
    assert not audit._exempt(audit.PACKAGE_ROOT / "periphery" / "predict.py")

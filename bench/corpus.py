"""Corpus builder: Pernix memories + hyperkb -> addressable prose entries.

Both stores use the same on-disk shape: a markdown file per topic, entries
separated by `---`, metadata carried in HTML comments (`<!-- @epoch: ... -->`).
Each entry becomes one evidence-tier document with a stable id, which is what
the spec's span anchors `(entry_id, content_hash, offset)` need (§3.2).

One substantive transform happens here: **`@weight` is renamed `@salience`**.
The real corpus carries `@weight: high` on ranked entries, and §3.2 renames
exactly that tag so a retrieval-ranking input can never be confused for a
committed number. The lexical firewall (§6.2) fails on this corpus without the
rename, which is the firewall behaving correctly on real data.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator

import os
PERNIX_MEMORIES = Path(os.environ.get("CANDOR_CORPUS_A", "corpus_sources/memories"))
HKB_STORAGE = Path(os.environ.get("CANDOR_CORPUS_B", "corpus_sources/kb"))

META = re.compile(r"<!--\s*@(\w+):\s*(.*?)\s*-->")
MIN_CHARS = 240          # below this an entry cannot carry a real question
MAX_CHARS = 6000         # keep prompts inside a sane budget


@dataclass
class Entry:
    entry_id: str
    source: str                     # 'pernix' | 'hkb'
    file: str
    text: str
    meta: dict[str, str] = field(default_factory=dict)

    @property
    def content_hash(self) -> str:
        return hashlib.sha256(self.text.encode("utf-8")).hexdigest()[:32]

    def to_json(self) -> dict[str, Any]:
        out = asdict(self)
        out["content_hash"] = self.content_hash
        return out

    def as_evidence_markdown(self) -> str:
        """Evidence-tier rendering. @weight never survives this boundary (§3.2)."""
        lines = [f"@entry_id: {self.entry_id}", f"@source: {self.source}"]
        for key, value in sorted(self.meta.items()):
            if key == "weight":
                key = "salience"
                value = {"high": "0.9", "normal": "0.5", "low": "0.2"}.get(value, "0.5")
            elif key in ("epoch", "type", "tags"):
                pass
            else:
                continue
            lines.append(f"@{key}: {value}")
        return "\n".join(lines) + "\n\n" + self.text + "\n"


def _split_entries(text: str) -> Iterator[tuple[dict[str, str], str]]:
    blocks = re.split(r"^---\s*$", text, flags=re.MULTILINE)
    for block in blocks:
        meta = {k.lower(): v for k, v in META.findall(block)}
        body = META.sub("", block).strip()
        if body:
            yield meta, body


def load_dir(directory: Path, source: str) -> list[Entry]:
    entries: list[Entry] = []
    if not directory.exists():
        return entries
    for path in sorted(directory.glob("*.md")):
        try:
            raw = path.read_text(encoding="utf-8", errors="replace")
        except OSError:
            continue
        for i, (meta, body) in enumerate(_split_entries(raw)):
            if not (MIN_CHARS <= len(body) <= MAX_CHARS):
                continue
            stamp = meta.get("epoch") or f"i{i}"
            entries.append(Entry(
                entry_id=f"{source}:{path.stem}#{stamp}",
                source=source, file=path.stem, text=body, meta=meta))
    return entries


def build(sources: Iterable[tuple[Path, str]] = (
        (PERNIX_MEMORIES, "pernix"), (HKB_STORAGE, "hkb"))) -> list[Entry]:
    """Whole corpus, de-duplicated by content hash and stably ordered."""
    seen: set[str] = set()
    out: list[Entry] = []
    for directory, name in sources:
        for entry in load_dir(directory, name):
            if entry.content_hash in seen:
                continue
            seen.add(entry.content_hash)
            out.append(entry)
    out.sort(key=lambda e: e.entry_id)
    return out


def save(entries: list[Entry], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for entry in entries:
            fh.write(json.dumps(entry.to_json(), ensure_ascii=False) + "\n")


def load(path: Path) -> list[Entry]:
    out: list[Entry] = []
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            obj = json.loads(line)
            obj.pop("content_hash", None)
            out.append(Entry(**obj))
    return out


def write_evidence_tier(entries: list[Entry], directory: Path) -> None:
    """Materialize the evidence tier CANDOR will retrieve over."""
    directory.mkdir(parents=True, exist_ok=True)
    for entry in entries:
        safe = re.sub(r"[^A-Za-z0-9._-]", "_", entry.entry_id)
        (directory / f"{safe}.md").write_text(
            entry.as_evidence_markdown(), encoding="utf-8")


if __name__ == "__main__":
    import sys
    entries = build()
    by_source: dict[str, int] = {}
    chars = 0
    for e in entries:
        by_source[e.source] = by_source.get(e.source, 0) + 1
        chars += len(e.text)
    print(f"entries: {len(entries)}  chars: {chars:,}  by source: {by_source}")
    if len(sys.argv) > 1:
        save(entries, Path(sys.argv[1]))
        print(f"wrote {sys.argv[1]}")

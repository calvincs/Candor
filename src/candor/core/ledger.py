"""The ledger: append-only JSONL segments + content-addressed payload store.

Trusted (spec §3.1, invariant I1). This module is the only primary artifact in
the system; every other store is a materialized view that can be deleted and
rebuilt from here.

Physical form
-------------
    <root>/segments/000001.jsonl   hash-chained event skeletons
    <root>/payloads/<hash>.json    content-addressed payloads (deletable)

The chain covers payload *commitments* only, so deleting a payload (redaction,
§3.1) leaves the chain verifying.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator, Optional

from .hashing import GENESIS, canon_json, hash_obj, sha256_hex

SEGMENT_LINES = 4096

# A CAS key is exactly a lowercase sha256 hex digest (what hash_obj produces).
# Anything else must never be interpolated into `cas_dir / f"{h}.json"`: a value
# containing `..` would escape the payload directory (M4 path traversal).
_PAYLOAD_HASH_RE = re.compile(r"^[0-9a-f]{64}$")


def is_payload_hash(value: Any) -> bool:
    return isinstance(value, str) and _PAYLOAD_HASH_RE.match(value) is not None

# Kinds admitted to the chain (spec §2 `events.kind` CHECK constraint).
EVENT_KINDS = frozenset({
    "assertion", "observation", "supersede", "admission", "demotion",
    "pin", "claim", "resolution", "alias", "redaction", "retraction",
    "checkpoint", "reliability",
})

# §3.1 durability: structural events fsync immediately, observations may batch.
# A reliability override is an operator lever on committed numbers — durable.
DURABLE_KINDS = frozenset({
    "admission", "resolution", "pin", "supersede", "alias", "redaction",
    "retraction", "checkpoint", "reliability",
})


@dataclass(frozen=True)
class Event:
    seq: int
    ts: int
    kind: str
    actor: str
    payload_hash: str
    source_ref: Optional[str]
    context_sig: Optional[str]
    prev_hash: str
    hash: str

    def as_row(self) -> dict[str, Any]:
        return {
            "seq": self.seq, "ts": self.ts, "kind": self.kind, "actor": self.actor,
            "payload_hash": self.payload_hash, "source_ref": self.source_ref,
            "context_sig": self.context_sig, "prev_hash": self.prev_hash,
            "hash": self.hash,
        }


def _event_hash(seq: int, ts: int, kind: str, actor: str, payload_hash: str,
                source_ref: Optional[str], context_sig: Optional[str],
                prev_hash: str) -> str:
    return sha256_hex(canon_json(
        [seq, ts, kind, actor, payload_hash, source_ref, context_sig, prev_hash]))


class LedgerError(RuntimeError):
    pass


class Ledger:
    """Append-only segments plus CAS. Single sequencer, single writer (spec §7)."""

    def __init__(self, root: Path, fsync_batch: int = 32, fsync_ms: int = 500) -> None:
        self.root = Path(root)
        self.seg_dir = self.root / "segments"
        self.cas_dir = self.root / "payloads"
        self.fsync_batch = fsync_batch
        self.fsync_ms = fsync_ms
        self._head = GENESIS
        self._seq = 0
        self._pending = 0
        self._last_sync = time.monotonic()
        self._fh = None
        self._fh_path: Optional[Path] = None
        self._lines_in_seg = 0

    # ── lifecycle ────────────────────────────────────────────────────────────
    def open(self) -> None:
        self.seg_dir.mkdir(parents=True, exist_ok=True)
        self.cas_dir.mkdir(parents=True, exist_ok=True)
        self._recover()

    def close(self) -> None:
        if self._fh is not None:
            self._flush(force=True)
            self._fh.close()
            self._fh = None
            self._fh_path = None

    def destroy(self) -> None:
        """Wipe the store completely (test-only reset path)."""
        self.close()
        for d in (self.seg_dir, self.cas_dir):
            if d.exists():
                for p in sorted(d.iterdir()):
                    p.unlink()
        self._head, self._seq, self._lines_in_seg = GENESIS, 0, 0

    # ── recovery (§3.1 torn-write) ───────────────────────────────────────────
    def _segments(self) -> list[Path]:
        return sorted(self.seg_dir.glob("*.jsonl"))

    def _recover(self) -> None:
        """Truncate the tail to the last line whose hash verifies. Never silent."""
        prev = GENESIS
        seq = 0
        segments = self._segments()
        truncating = False
        for i, seg in enumerate(segments):
            if truncating:
                seg.unlink()
                continue
            keep = bytearray()
            with seg.open("rb") as fh:
                raw = fh.read()
            for line in raw.splitlines(keepends=True):
                stripped = line.strip()
                if not stripped:
                    truncating = True
                    break
                try:
                    rec = json.loads(stripped)
                except Exception:
                    truncating = True
                    break
                ok = (
                    isinstance(rec, dict)
                    and rec.get("prev_hash") == prev
                    and rec.get("seq") == seq + 1
                    and rec.get("hash") == _event_hash(
                        rec.get("seq"), rec.get("ts"), rec.get("kind"), rec.get("actor"),
                        rec.get("payload_hash"), rec.get("source_ref"),
                        rec.get("context_sig"), rec.get("prev_hash"))
                )
                if not ok:
                    truncating = True
                    break
                keep += line if line.endswith(b"\n") else line + b"\n"
                prev = rec["hash"]
                seq = rec["seq"]
            if truncating:
                if keep:
                    seg.write_bytes(bytes(keep))
                else:
                    seg.unlink()
        self._head = prev
        self._seq = seq
        self._lines_in_seg = 0
        segs = self._segments()
        if segs:
            self._lines_in_seg = sum(1 for _ in segs[-1].open("rb"))

    # ── append ───────────────────────────────────────────────────────────────
    def _open_tail(self):
        segs = self._segments()
        if not segs or self._lines_in_seg >= SEGMENT_LINES:
            idx = (int(segs[-1].stem) + 1) if segs else 1
            path = self.seg_dir / f"{idx:06d}.jsonl"
            path.touch()
            self._lines_in_seg = 0
        else:
            path = segs[-1]
        if self._fh_path != path:
            if self._fh is not None:
                self._fh.flush()
                os.fsync(self._fh.fileno())
                self._fh.close()
            self._fh = path.open("ab")
            self._fh_path = path
        return self._fh

    def _flush(self, force: bool) -> None:
        if self._fh is None:
            return
        self._fh.flush()
        now = time.monotonic()
        if (force or self._pending >= self.fsync_batch
                or (now - self._last_sync) * 1000.0 >= self.fsync_ms):
            os.fsync(self._fh.fileno())
            self._pending = 0
            self._last_sync = now

    def put_payload(self, payload: Any) -> str:
        h = hash_obj(payload)
        path = self.cas_dir / f"{h}.json"
        if not path.exists():
            tmp = path.with_suffix(".json.tmp")
            tmp.write_text(canon_json(payload), encoding="utf-8")
            tmp.replace(path)
        return h

    def append(self, kind: str, actor: str, payload: Any,
               source_ref: Optional[str] = None,
               context_sig: Optional[str] = None,
               ts: Optional[int] = None) -> Event:
        if kind not in EVENT_KINDS:
            raise LedgerError(f"event kind not in the chain's CHECK set: {kind!r}")
        payload_hash = self.put_payload(payload)
        seq = self._seq + 1
        ts = int(time.time() * 1000) if ts is None else ts
        h = _event_hash(seq, ts, kind, actor, payload_hash, source_ref,
                        context_sig, self._head)
        ev = Event(seq, ts, kind, actor, payload_hash, source_ref, context_sig,
                   self._head, h)
        fh = self._open_tail()
        fh.write((canon_json(ev.as_row()) + "\n").encode("utf-8"))
        self._lines_in_seg += 1
        self._pending += 1
        self._flush(force=kind in DURABLE_KINDS)
        self._head = h
        self._seq = seq
        return ev

    # ── read ─────────────────────────────────────────────────────────────────
    def head(self) -> str:
        return self._head

    def seq(self) -> int:
        return self._seq

    def read_all(self) -> Iterator[Event]:
        if self._fh is not None:
            self._flush(force=True)
        for seg in self._segments():
            with seg.open("rb") as fh:
                for line in fh:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    yield Event(rec["seq"], rec["ts"], rec["kind"], rec["actor"],
                                rec["payload_hash"], rec["source_ref"],
                                rec["context_sig"], rec["prev_hash"], rec["hash"])

    def payload(self, payload_hash: str) -> Optional[Any]:
        # Refuse to interpolate anything but a real digest into the CAS path
        # (M4): a non-hash cannot name a stored payload anyway, so a safe no-op.
        if not is_payload_hash(payload_hash):
            return None
        path = self.cas_dir / f"{payload_hash}.json"
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def delete_payload(self, payload_hash: str) -> bool:
        # Defense in depth (M4): never unlink a path built from a non-digest,
        # which could contain `..` and escape cas_dir. A no-op, cannot escape.
        if not is_payload_hash(payload_hash):
            return False
        path = self.cas_dir / f"{payload_hash}.json"
        if path.exists():
            path.unlink()
            return True
        return False

    def verify_chain(self) -> bool:
        prev, seq = GENESIS, 0
        for ev in self.read_all():
            if ev.prev_hash != prev or ev.seq != seq + 1:
                return False
            if ev.hash != _event_hash(ev.seq, ev.ts, ev.kind, ev.actor,
                                      ev.payload_hash, ev.source_ref,
                                      ev.context_sig, ev.prev_hash):
                return False
            prev, seq = ev.hash, ev.seq
        return prev == self._head

    # ── test-only fault injection support (§6.1) ─────────────────────────────
    def append_raw_line(self, text: str) -> None:
        """Write an unverifiable line to the tail segment (torn-write simulation)."""
        fh = self._open_tail()
        fh.write(text.encode("utf-8"))
        fh.flush()
        os.fsync(fh.fileno())
        self._fh.close()
        self._fh = None
        self._fh_path = None

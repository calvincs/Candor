"""Pernix memories -> CANDOR observation streams (the real-world bridge).

Not a benchmark: an exploration of whether the curiosity engine finds true
structure in lived agent history. The LLM proposes structured observation
events from dated memory entries; mechanical filters dispose (fixed predicate
vocabulary, required fields, sane epochs, bounded ctx). Streams are written
sorted by epoch so ingestion order = historical order.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from . import ollama
from .corpus import META, _split_entries

LIVE = Path("data/bench/pernix_live")
OUT = Path("data/bench/pernix_obs.jsonl")

# explicit outcome signals only — generic fetch/scrape verbs match everything
OUTCOMEISH = re.compile(
    r"\b(403|404|401|429|500|fail(ed|ure)?|succe(ss|ssfully|eded)|blocked|"
    r"timeout|no results|denied|paywall|cloudflare|bot[- ](wall|detection|"
    r"protection))\b", re.I)

SYSTEM = (
    "You convert an agent's dated memory note into observation events. The "
    "note is DISTILLED real experience: a stated capability, failure, or "
    "reliability judgement counts as an observed attempt (e.g. 'site X "
    "returns 403 via direct fetch' -> fetch_ok(X)=false; 'aggregator Y is "
    "reliable for headlines' -> fetch_ok(Y)=true). Report only what the note "
    "asserts from experience; never invent targets. JSON array only.")

PROMPT = """NOTE (epoch {epoch}):
\"\"\"
{text}
\"\"\"

Extract observation events as a JSON array. Each event:
  {{"pred": "fetch_ok" | "tool_ok" | "search_ok",
    "arg": "<domain like rrstar.com | tool/pipeline name | engine name>",
    "outcome": true|false,
    "ctx": {{"method": "browse|http|crawl4ai|search|cache|api|other",
            "protection": "cloudflare|bot_wall|paywall|js_render|none|unknown",
            "kind": "news|docs|social|gov|finance|code|media|other"}}}}

fetch_ok = a web fetch/scrape attempt; tool_ok = a tool or pipeline run;
search_ok = a search-engine retrieval attempt. outcome true = it worked.
One event per distinct target+method experience the note asserts. Use the
specific domain/tool named. [] only if the note contains no experience-backed
success or failure at all. JSON only."""

VALID_PRED = {"fetch_ok", "tool_ok", "search_ok"}
VALID_METHOD = {"browse", "http", "crawl4ai", "search", "cache", "api", "other"}
VALID_PROT = {"cloudflare", "bot_wall", "paywall", "js_render", "none", "unknown"}
VALID_KIND = {"news", "docs", "social", "gov", "finance", "code", "media", "other"}
ARG = re.compile(r"^[a-z0-9][a-z0-9_.\-]{1,60}$")


def entries_with_epochs() -> list[tuple[int, str]]:
    out = []
    for path in sorted(LIVE.glob("*.md")):
        for meta, body in _split_entries(path.read_text(encoding="utf-8",
                                                        errors="replace")):
            epoch = meta.get("epoch")
            if epoch and epoch.isdigit() and 240 <= len(body) <= 6000 \
                    and OUTCOMEISH.search(body):
                out.append((int(epoch), body))
    out.sort()
    return out


def extract(item: tuple[int, str]) -> list[dict]:
    epoch, text = item
    raw = ollama.generate(PROMPT.format(epoch=epoch, text=text[:4000]),
                          system=SYSTEM, num_predict=700)
    parsed = ollama.extract_json(raw)
    events = []
    if isinstance(parsed, list):
        for ev in parsed[:12]:
            if not isinstance(ev, dict) or ev.get("pred") not in VALID_PRED:
                continue
            arg = str(ev.get("arg", "")).strip().lower()
            if not ARG.match(arg) or not isinstance(ev.get("outcome"), bool):
                continue
            ctx = ev.get("ctx") or {}
            events.append({
                "pred": ev["pred"], "arg": arg, "outcome": ev["outcome"],
                "epoch": epoch,
                "ctx": {
                    "method": ctx.get("method") if ctx.get("method") in VALID_METHOD else "other",
                    "protection": ctx.get("protection") if ctx.get("protection") in VALID_PROT else "unknown",
                    "kind": ctx.get("kind") if ctx.get("kind") in VALID_KIND else "other",
                }})
    return events


def main() -> None:
    items = entries_with_epochs()
    print(f"outcome-ish dated entries: {len(items)}")
    results = ollama.parallel(
        extract, items, workers=1,
        on_progress=lambda d, t: (d % 50 == 0 or d == t) and
        print(f"  extracted {d}/{t}", flush=True))
    seen, events = set(), []
    for batch in results:
        if not isinstance(batch, list):
            continue
        for ev in batch:
            key = (ev["pred"], ev["arg"], ev["outcome"], ev["epoch"])
            if key not in seen:
                seen.add(key)
                events.append(ev)
    events.sort(key=lambda e: e["epoch"])
    OUT.write_text("\n".join(json.dumps(e) for e in events), encoding="utf-8")
    from collections import Counter
    print(f"events: {len(events)}  by pred: "
          f"{dict(Counter(e['pred'] for e in events))}")
    print(f"distinct args: {len({(e['pred'], e['arg']) for e in events})}")


if __name__ == "__main__":
    main()

"""Thin Ollama client for the §6.8 harness: generate + embed, cached on disk.

Cached because the honest test must be re-runnable without re-rolling the
model. The cache key covers model, prompt, and every sampling option, so a
changed prompt is a cache miss rather than a silent stale hit. Temperature is 0
throughout: the generated suite has to be reproducible or the pre-registration
means nothing.
"""

from __future__ import annotations

import json
import hashlib
import threading
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import os
# The GPU host serving generation + embeddings. The research runs used an
# internal LAN box; point this at any ollama-compatible endpoint.
HOST = os.environ.get("CANDOR_BENCH_OLLAMA", "http://localhost:11434")

# Two roles, deliberately split (PREREGISTRATION.md, Amendment 1).
#
# GEN_MODEL authors the suite. Its quality is a *throughput* concern, not a
# correctness one: every item it proposes is verified against the corpus by
# exact string match, so a weaker author lowers the yield of valid items and
# can never produce an invalid label. laguna-s-2.1 is a 117B dense model at
# ~14 tok/s here; the MoE below runs the same prompt in 1.5s against ~100s.
#
# JUDGE_MODEL is the baseline's probability elicitor and the large observer.
# That one has to be strong, or §6.8 is measured against a strawman.
GEN_MODEL = "qwen3.6:35b-a3b-q8_0"
JUDGE_MODEL = "qwen3.6:27b-q8_0"
LEGACY_JUDGE_MODEL = "laguna-s-2.1:q8_0"   # Amendment 3: agreement is reported
EMBED_MODEL = "bge-m3:latest"
CACHE_DIR = Path("data/bench/cache")
KEEP_ALIVE = "30m"


class OllamaError(RuntimeError):
    pass


class FieldRejected(OllamaError):
    """The server refused a request field (HTTP 400), e.g. `think` on a
    model that has no thinking mode. Distinct from a timeout, because the
    remedy is different and must not be applied to the wrong failure."""


def _post(path: str, payload: dict[str, Any], timeout: float) -> dict[str, Any]:
    req = urllib.request.Request(
        HOST + path, data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        if exc.code == 400:
            raise FieldRejected(f"{path}: {exc}") from exc
        raise OllamaError(f"{path}: {exc}") from exc
    except urllib.error.URLError as exc:
        raise OllamaError(f"{path}: {exc}") from exc


class Cache:
    def __init__(self, directory: Path = CACHE_DIR) -> None:
        self.dir = Path(directory)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _path(self, key: str) -> Path:
        return self.dir / f"{key}.json"

    def get(self, key: str) -> Optional[Any]:
        path = self._path(key)
        if not path.exists():
            with self._lock:
                self.misses += 1
            return None
        with self._lock:
            self.hits += 1
        return json.loads(path.read_text(encoding="utf-8"))

    def put(self, key: str, value: Any) -> None:
        tmp = self._path(key).with_suffix(".tmp")
        tmp.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")
        tmp.replace(self._path(key))


_CACHE = Cache()


def _key(*parts: Any) -> str:
    blob = json.dumps(parts, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:40]


def generate(prompt: str, model: str = GEN_MODEL, num_predict: int = 512,
             temperature: float = 0.0, system: Optional[str] = None,
             timeout: float = 90.0, cache: Cache = _CACHE,
             attempts: int = 3) -> str:
    """One cached generation.

    The timeout is deliberately short with retries rather than long without.
    Individual requests to this server occasionally never return; with a 600s
    timeout and a small worker pool, two such requests stall a whole phase for
    ten minutes. A 300s ceiling turns that into a retry.
    """
    key = _key("gen", model, prompt, system, num_predict, temperature)
    hit = cache.get(key)
    if hit is not None:
        return hit
    payload: dict[str, Any] = {
        "model": model, "prompt": prompt, "stream": False, "think": False,
        "keep_alive": KEEP_ALIVE,
        "options": {"num_predict": num_predict, "temperature": temperature,
                    "top_p": 1.0, "seed": 7},
    }
    if system:
        payload["system"] = system
    last: Optional[Exception] = None
    for _ in range(attempts):
        try:
            out = _post("/api/generate", payload, timeout)
            text = out.get("response", "")
            cache.put(key, text)
            return text
        except FieldRejected as exc:
            # Only a model with no thinking mode at all may drop the field.
            # Dropping it on a timeout would silently re-enable thinking, which
            # is the opposite of what we want: thinking tokens are pure latency
            # here, since every judgement is a single boolean or probability.
            last = exc
            if "think" in payload:
                payload.pop("think")
            else:
                break
        except OllamaError as exc:
            last = exc          # timeout or transport error: retry unchanged
            print(f"    [retry] {model}: {exc}", flush=True)
    raise OllamaError(f"generate failed after {attempts} attempts: {last}")


def embed(texts: Sequence[str], model: str = EMBED_MODEL,
          timeout: float = 600.0, cache: Cache = _CACHE) -> list[list[float]]:
    """Embed a batch, reusing per-text cache entries."""
    out: list[Optional[list[float]]] = [None] * len(texts)
    pending: list[int] = []
    for i, text in enumerate(texts):
        hit = cache.get(_key("embed", model, text))
        if hit is None:
            pending.append(i)
        else:
            out[i] = hit
    for start in range(0, len(pending), 32):
        chunk = pending[start:start + 32]
        resp = _post("/api/embed",
                     {"model": model, "input": [texts[i] for i in chunk],
                      "keep_alive": KEEP_ALIVE}, timeout)
        vectors = resp.get("embeddings") or []
        if len(vectors) != len(chunk):
            raise OllamaError(f"embed returned {len(vectors)} for {len(chunk)} inputs")
        for i, vec in zip(chunk, vectors):
            out[i] = vec
            cache.put(_key("embed", model, texts[i]), vec)
    return [v for v in out if v is not None]


def parallel(fn: Callable[[Any], Any], items: Sequence[Any], workers: int = 1,
             on_progress: Optional[Callable[[int, int], None]] = None) -> list[Any]:
    """Map over items. **Serial by default, and that is not an oversight.**

    This ollama server executes one request at a time; concurrent callers
    simply queue behind each other. A worker pool therefore buys nothing and
    costs a great deal: each queued request ages toward the client timeout
    while it waits, times out, retries, and adds further queue pressure — a
    feedback loop that presents as "fast at first, then stalls for ten
    minutes". Raise `workers` only against a backend that genuinely runs
    requests in parallel.
    """
    results: list[Any] = [None] * len(items)
    done = 0
    lock = threading.Lock()

    def run(idx_item):
        nonlocal done
        idx, item = idx_item
        try:
            value = fn(item)
        except Exception as exc:                      # noqa: BLE001
            value = {"__error__": f"{type(exc).__name__}: {exc}"}
        results[idx] = value
        with lock:
            done += 1
            if on_progress:
                on_progress(done, len(items))
        return None

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(run, enumerate(items)))
    return results


def extract_json(text: str) -> Optional[Any]:
    """Pull the first JSON object/array out of a model response."""
    text = text.strip()
    if text.startswith("```"):
        text = text.split("```")[1]
        if text.startswith("json"):
            text = text[4:]
    # honor whichever bracket opens first — an array of objects must parse as
    # the array, not as its first element
    starts = [(text.find(o), o, c) for o, c in (("{", "}"), ("[", "]"))
              if text.find(o) >= 0]
    for start, opener, closer in sorted(starts):
        depth = 0
        in_str = False
        escape = False
        for i in range(start, len(text)):
            ch = text[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == opener:
                depth += 1
            elif ch == closer:
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i + 1])
                    except ValueError:
                        break
    return None


def health() -> dict[str, Any]:
    req = urllib.request.Request(HOST + "/api/tags")
    with urllib.request.urlopen(req, timeout=15) as resp:
        tags = json.loads(resp.read().decode("utf-8"))
    names = {m["name"] for m in tags.get("models", [])}
    return {"host": HOST, "gen_model": GEN_MODEL, "embed_model": EMBED_MODEL,
            "gen_available": GEN_MODEL in names,
            "embed_available": EMBED_MODEL in names}

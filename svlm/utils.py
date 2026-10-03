"""Small shared helpers: JSONL I/O, resumable shards, hashing, logging, timing."""
from __future__ import annotations

import hashlib
import json
import logging
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterable, Iterator

log = logging.getLogger("svlm")
if not log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s", "%H:%M:%S"))
    log.addHandler(_h)
    log.setLevel(logging.INFO)


def read_jsonl(path: str | Path) -> list[dict]:
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def iter_jsonl(path: str | Path) -> Iterator[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: str | Path, rows: Iterable[dict]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    tmp.replace(path)  # atomic: a disconnect never leaves a half-written file
    return n


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_id(*parts: object) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


def chunks(seq: list, size: int) -> Iterator[list]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def run_sharded(
    items: list[dict],
    out_dir: Path,
    fn: Callable[[list[dict]], list[dict]],
    shard_size: int = 200,
    desc: str = "",
) -> list[dict]:
    """Process `items` in shards, writing shard_XXXXX.jsonl as each finishes.

    Finished shards are skipped on restart, so a Colab disconnect costs at most one shard.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    shards = list(chunks(items, shard_size))
    results: list[dict] = []
    for i, shard in enumerate(shards):
        path = out_dir / f"shard_{i:05d}.jsonl"
        if path.exists():
            results.extend(read_jsonl(path))
            continue
        t0 = time.time()
        out = fn(shard)
        write_jsonl(path, out)
        results.extend(out)
        log.info("%s shard %d/%d done in %.0fs", desc, i + 1, len(shards), time.time() - t0)
    return results


@contextmanager
def timer(label: str):
    t0 = time.time()
    yield
    log.info("%s took %.1fs", label, time.time() - t0)


def write_json(path: str | Path, obj: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))

"""Step 1 — prepare_inputs.

Collect source files from pinned, permissively licensed repositories, build FIM samples,
remove overlap with evaluation benchmarks, and split train / held-out by *file* so no held-out
file is ever seen in training.

Outputs (in runs/.../prepare/):
    files.jsonl     one row per source file (content, repo, ref, licence)
    train.jsonl     FIM samples for training
    heldout.jsonl   FIM samples for evaluation
    manifest.json   counts, licences, commit hashes, data hashes
"""
from __future__ import annotations

import fnmatch
import random
import subprocess
from pathlib import Path

from .. import decontam
from ..config import Run
from ..fim import build_samples
from ..utils import log, sha256_file, stable_id, write_json, write_jsonl

STEP = "prepare"


def _clone(repo: str, ref: str, dest: Path) -> str:
    if not (dest / ".git").exists():
        dest.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", "--depth", "1", "--branch", ref, repo, str(dest)], check=True)
    sha = subprocess.run(["git", "-C", str(dest), "rev-parse", "HEAD"], check=True, capture_output=True, text=True)
    return sha.stdout.strip()


def _matches(rel: str, patterns: list[str]) -> bool:
    return any(fnmatch.fnmatch(rel, p) for p in patterns)


def collect_files(src: dict, checkout: Path, exclude: list[str], max_bytes: int) -> list[dict]:
    out = []
    for path in sorted(checkout.rglob("*.py")):
        rel = path.relative_to(checkout).as_posix()
        if not _matches(rel, src["include"]) or _matches(rel, exclude):
            continue
        if path.stat().st_size > max_bytes:
            continue
        try:
            content = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        out.append(
            {
                "file_id": stable_id(src["repo"], src["ref"], rel),
                "repo": src["repo"],
                "ref": src["ref"],
                "license": src["license"],
                "path": rel,
                "content": content,
            }
        )
    return out


def run(run: Run) -> dict:
    cfg = run.cfg
    d = cfg["data"]
    out = run.step_dir(STEP)
    cache = run.root / "cache" / "repos"

    files: list[dict] = []
    commits = {}
    for src in d["sources"]:
        name = src["repo"].rstrip("/").split("/")[-1]
        sha = _clone(src["repo"], src["ref"], cache / f"{name}-{src['ref']}")
        commits[src["repo"]] = {"ref": src["ref"], "commit": sha, "license": src["license"]}
        got = collect_files(src, cache / f"{name}-{src['ref']}", d.get("exclude", []), d.get("max_file_bytes", 80000))
        log.info("%s@%s: %d files", name, src["ref"], len(got))
        files.extend(got)
    if not files:
        raise RuntimeError("no source files collected; check data.sources include patterns")

    # split by file, so held-out code is never seen in training
    rng = random.Random(cfg.get("seed", 42))
    rng.shuffle(files)
    fim = d["fim"]
    n_train, n_held = fim["n_train"], fim["n_heldout"]
    held_share = n_held / (n_train + n_held)
    n_held_files = max(1, int(len(files) * held_share))
    held_files, train_files = files[:n_held_files], files[n_held_files:]

    common = dict(
        span_mix=fim["span_mix"],
        max_middle_lines=fim["max_middle_lines"],
        max_prefix_chars=fim["max_prefix_chars"],
        max_suffix_chars=fim["max_suffix_chars"],
        seed=cfg.get("seed", 42),
    )
    # draw extra, decontamination removes some
    train = build_samples(train_files, int(n_train * 1.1), **common)
    held = build_samples(held_files, int(n_held * 1.1), **common)

    dc = d.get("decontam", {})
    n = dc.get("ngram", 13)
    index = decontam.build_index(decontam.load_eval_texts(dc.get("eval_sets", [])), n)

    def clean(rows):
        keep = [r for r in rows if not decontam.is_contaminated(r["prefix"] + r["middle"] + r["suffix"], index, n)]
        return keep, len(rows) - len(keep)

    train, drop_t = clean(train)
    held, drop_h = clean(held)
    train, held = train[:n_train], held[:n_held]

    write_jsonl(out / "files.jsonl", files)
    write_jsonl(out / "train.jsonl", [{**r, "split": "train"} for r in train])
    write_jsonl(out / "heldout.jsonl", [{**r, "split": "heldout"} for r in held])

    span_counts = {}
    for r in train:
        span_counts[r["span_type"]] = span_counts.get(r["span_type"], 0) + 1
    manifest = {
        "sources": commits,
        "n_files": len(files),
        "n_train_files": len(train_files),
        "n_heldout_files": len(held_files),
        "n_train": len(train),
        "n_heldout": len(held),
        "train_span_counts": span_counts,
        "decontam_dropped": {"train": drop_t, "heldout": drop_h, "ngram": n, "index_size": len(index)},
        "sha256": {
            "train.jsonl": sha256_file(out / "train.jsonl"),
            "heldout.jsonl": sha256_file(out / "heldout.jsonl"),
        },
    }
    write_json(out / "manifest.json", manifest)
    log.info("prepare: %d train / %d held-out samples from %d files", len(train), len(held), len(files))
    return manifest

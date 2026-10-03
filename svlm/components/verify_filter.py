"""Step 3 — verify_filter.

* Checks teacher completions on train samples: keep only those that parse when spliced back
  into the full file AND stay close to the original code (edit similarity >= accept_similarity).
* Builds the final SFT set: every original span + accepted teacher spans.
* Scores the teacher on the held-out set (the ceiling for the release gates).

Outputs: sft.jsonl, teacher_heldout_scores.jsonl, filter_report.json
"""
from __future__ import annotations

from ..config import Run
from ..fim import parses_in_file, score
from ..utils import iter_jsonl, log, read_jsonl, sha256_file, write_json, write_jsonl

STEP = "verify"


def _load_shards(d) -> list[dict]:
    rows: list[dict] = []
    if d.exists():
        for p in sorted(d.glob("shard_*.jsonl")):
            rows.extend(read_jsonl(p))
    return rows


def _mean(rows: list[dict], key: str) -> float:
    return sum(r[key] for r in rows) / len(rows) if rows else 0.0


def run(run: Run) -> dict:
    cfg = run.cfg
    prep, gen_dir, out = run.step_dir("prepare"), run.step_dir("teacher_generate"), run.step_dir(STEP)
    files = {f["file_id"]: f["content"] for f in iter_jsonl(prep / "files.jsonl")}
    train = {r["id"]: r for r in read_jsonl(prep / "train.jsonl")}
    held = {r["id"]: r for r in read_jsonl(prep / "heldout.jsonl")}
    thr = cfg["data"].get("accept_similarity", 0.5)

    reasons = {"accepted": 0, "empty": 0, "does_not_parse": 0, "too_different": 0}
    sft = [{**r, "target": r["middle"], "source": "original"} for r in train.values()]
    for g in _load_shards(gen_dir / "train"):
        r = train.get(g["id"])
        if r is None:
            continue
        comp = g["teacher_completion"]
        if not comp.strip():
            reasons["empty"] += 1
            continue
        if not parses_in_file(files[r["file_id"]], r["start"], r["end"], comp):
            reasons["does_not_parse"] += 1
            continue
        s = score(comp, r)
        if s["exact_match"] == 1.0:
            reasons["accepted"] += 1  # identical to original: already in the set
            continue
        if s["edit_similarity"] < thr:
            reasons["too_different"] += 1
            continue
        sft.append({**r, "target": comp, "source": "teacher"})
        reasons["accepted"] += 1

    scored = []
    for g in _load_shards(gen_dir / "heldout"):
        r = held.get(g["id"])
        if r is None:
            continue
        s = score(g["teacher_completion"], r, files[r["file_id"]])
        scored.append({"id": r["id"], "span_type": r["span_type"], "pred": g["teacher_completion"], **s})

    write_jsonl(out / "sft.jsonl", sft)
    write_jsonl(out / "teacher_heldout_scores.jsonl", scored)
    report = {
        "teacher_train_filter": reasons,
        "n_sft": len(sft),
        "n_sft_teacher": sum(1 for r in sft if r["source"] == "teacher"),
        "teacher_heldout": {
            "n": len(scored),
            "exact_match": _mean(scored, "exact_match"),
            "edit_similarity": _mean(scored, "edit_similarity"),
            "parse_rate": _mean(scored, "parses"),
        },
        "sha256": {"sft.jsonl": sha256_file(out / "sft.jsonl")},
    }
    write_json(out / "filter_report.json", report)
    log.info("verify: %s", report)
    return report

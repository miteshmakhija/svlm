"""Step 7 — evaluate: score base, SFT and final (KD) student on the held-out set, measure
latency, optionally run HumanEval+ and apply the release gates.

Output: runs/.../evaluate/metrics.json  (gate verdicts included), predictions_<variant>.jsonl
"""
from __future__ import annotations

import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from ..config import Run
from ..fim import fim_prompt, score, trim_completion
from ..modeling import free_gpu, generate_batch, load_student_for_inference, merge_and_save
from ..utils import chunks, iter_jsonl, log, read_json, read_jsonl, write_json, write_jsonl

STEP = "evaluate"


def _summarise(rows: list[dict]) -> dict:
    def mean(k, rs):
        return round(sum(r[k] for r in rs) / len(rs), 4) if rs else 0.0

    out = {"n": len(rows), "exact_match": mean("exact_match", rows), "edit_similarity": mean("edit_similarity", rows),
           "parse_rate": mean("parses", rows), "by_span": {}}
    for t in sorted({r["span_type"] for r in rows}):
        sub = [r for r in rows if r["span_type"] == t]
        out["by_span"][t] = {"n": len(sub), "exact_match": mean("exact_match", sub), "edit_similarity": mean("edit_similarity", sub)}
    return out


def eval_variant(cfg: dict, adapter: Path | None, held: list[dict], files: dict, out: Path, name: str) -> dict:
    pred_path = out / f"predictions_{name}.jsonl"
    if pred_path.exists():
        return _summarise(read_jsonl(pred_path))
    model, tok = load_student_for_inference(cfg, adapter)
    e = cfg["eval"]
    order = sorted(range(len(held)), key=lambda i: len(held[i]["prefix"]) + len(held[i]["suffix"]))
    preds: list[str] = [""] * len(held)
    for b in chunks(order, e.get("gen_batch_size", 32)):
        raw = generate_batch(model, tok, [fim_prompt(held[i]["prefix"], held[i]["suffix"]) for i in b], e["max_new_tokens"])
        for i, r in zip(b, raw):
            preds[i] = trim_completion(r, held[i]["span_type"], held[i]["suffix"])
    rows = [{"id": h["id"], "span_type": h["span_type"], "pred": p, **score(p, h, files[h["file_id"]])} for h, p in zip(held, preds)]
    write_jsonl(pred_path, rows)
    lat = measure_ttft(model, tok, held[: e.get("latency_prompts", 50)]) if name == "final" else None
    free_gpu(model)
    s = _summarise(rows)
    if lat:
        s["latency"] = lat
    return s


@torch.no_grad()
def measure_ttft(model, tok, rows: list[dict]) -> dict:
    """Time to first token, batch size 1, on this GPU (prefill + one decode step)."""
    times = []
    for i, r in enumerate(rows):
        enc = tok(fim_prompt(r["prefix"], r["suffix"]), return_tensors="pt", add_special_tokens=False).to(model.device)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        model.generate(**enc, max_new_tokens=1, do_sample=False, pad_token_id=tok.pad_token_id)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if i >= 3:  # skip warm-up
            times.append((time.perf_counter() - t0) * 1000)
    if not times:
        return {}
    times.sort()
    return {"ttft_ms_p50": round(statistics.median(times), 1), "ttft_ms_p90": round(times[int(0.9 * (len(times) - 1))], 1),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu"}


def run_humaneval(merged: Path, out: Path) -> dict:
    """HumanEval+ via evalplus (plain completion, greedy). Best effort: returns a reason on failure."""
    try:
        res = subprocess.run(
            [sys.executable, "-m", "evalplus.evaluate", "--model", str(merged), "--dataset", "humaneval",
             "--backend", "hf", "--greedy", "--root", str(out / "evalplus"), "--force-base-prompt"],
            capture_output=True, text=True, timeout=3600,
        )
        (out / "evalplus.log").write_text(res.stdout + "\n" + res.stderr)
        files = sorted((out / "evalplus").rglob("*eval_results.json"))
        if not files:
            return {"status": "skipped", "reason": "evalplus produced no results; see evalplus.log"}
        data = json.loads(files[-1].read_text())
        pass_at = data.get("pass_at_k", {})
        return {"status": "ok", "humaneval_pass@1": pass_at.get("base", {}).get("pass@1"),
                "humaneval_plus_pass@1": pass_at.get("plus", {}).get("pass@1")}
    except Exception as e:
        return {"status": "skipped", "reason": str(e)}


def apply_gates(cfg: dict, m: dict) -> dict:
    g = cfg.get("gates", {})
    final, base, teacher = m["final"], m["base"], m["teacher"]
    checks = {}
    thr = g.get("quality_vs_teacher", 0.85)
    if teacher.get("edit_similarity"):
        ratio = final["edit_similarity"] / teacher["edit_similarity"]
        checks["quality_vs_teacher"] = {"value": round(ratio, 3), "threshold": thr, "pass": ratio >= thr}
    else:  # no usable teacher score means the ceiling is unknown: never pass silently
        checks["quality_vs_teacher"] = {"value": "teacher score missing or zero", "threshold": thr, "pass": False}
    if g.get("beats_base", True):
        checks["beats_base"] = {"value": final["edit_similarity"], "base": base["edit_similarity"],
                                "pass": final["edit_similarity"] > base["edit_similarity"]}
    checks["parse_rate"] = {"value": final["parse_rate"], "threshold": g.get("min_parse_rate", 0.98),
                            "pass": final["parse_rate"] >= g.get("min_parse_rate", 0.98)}
    ttft = final.get("latency", {}).get("ttft_ms_p50")
    if ttft is not None and "max_ttft_ms_t4" in g:
        checks["ttft"] = {"value": ttft, "threshold": g["max_ttft_ms_t4"], "pass": ttft <= g["max_ttft_ms_t4"]}
    return {"checks": checks, "verdict": "pass" if all(c["pass"] for c in checks.values()) else "fail"}


def run(run: Run) -> dict:
    cfg = run.cfg
    out = run.step_dir(STEP)
    held = read_jsonl(run.step_dir("prepare") / "heldout.jsonl")
    files = {f["file_id"]: f["content"] for f in iter_jsonl(run.step_dir("prepare") / "files.jsonl")}
    sft = run.step_dir("train_sft") / "adapter"
    kd = run.step_dir("train_kd") / "adapter"
    final_adapter = kd if kd.exists() else sft

    metrics = {
        "teacher": read_json(run.step_dir("verify") / "filter_report.json")["teacher_heldout"],
        "base": eval_variant(cfg, None, held, files, out, "base"),
        "sft": eval_variant(cfg, sft, held, files, out, "sft") if sft.exists() else None,
        "final": eval_variant(cfg, final_adapter, held, files, out, "final"),
        "final_adapter": str(final_adapter),
    }
    if cfg["eval"].get("humaneval", False):
        merged = merge_and_save(cfg, final_adapter, run.run_dir / "merged")
        metrics["humaneval"] = run_humaneval(merged, out)
    metrics["gates"] = apply_gates(cfg, metrics)
    write_json(out / "metrics.json", metrics)
    log.info("evaluate: verdict=%s final=%s teacher=%s base=%s", metrics["gates"]["verdict"],
             metrics["final"]["edit_similarity"], metrics["teacher"]["edit_similarity"], metrics["base"]["edit_similarity"])
    return metrics

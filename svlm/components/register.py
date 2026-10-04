"""Step 9 — register: publish a passing model into the model store with a model card.

model_store/
  catalogue.yaml                       live version per model (the gateway reads this)
  <name>/<version>-<tier>/
      model.<type>.gguf   merged/   model_card.md   metrics.json   lineage.json

A model whose gates failed is not registered unless `register.force=true` is set.
"""
from __future__ import annotations

import datetime as dt
import shutil
import subprocess
from pathlib import Path

import yaml

from ..config import Run, config_hash
from ..utils import log, read_json, write_json

STEP = "register"
REPO_ROOT = Path(__file__).resolve().parents[2]

USE = {
    "code": ("One Python coder for the IDE: inline completion (fill-in-the-middle) and chat (write, explain, document "
             "and fix code).",
             "Long multi-file refactors, non-Python languages, anything outside code.",
             "Trained on public BSD/Apache Python, MBPP and teacher-written problems; no MSCI code yet. Chat answers are "
             "checked by tests in training only where tests exist. Inherits teacher mistakes on accepted answers."),
    "fim": ("Inline code completion inside the IDE (fill-in-the-middle), Python first.",
            "Chat, explanations, multi-file refactors (use msci-code-deep), anything outside code.",
            "Trained on public BSD/Apache Python only; no MSCI code yet. Short completions (<= 12 lines). "
            "Inherits teacher mistakes on spans where the teacher was accepted."),
}


SERVING = {
    "fim": "Recommended settings: temperature 0, max new tokens 64, stop on `<|endoftext|>` and FIM tokens.",
    "code": ("Recommended settings: inline completion as for msci-code-fast (FIM prompt, temperature 0, max new tokens 64); "
             "chat with ChatML and the system prompt in `svlm/chat.py`, temperature 0-0.2, stop on `<|im_end|>`."),
}


def _latency_lines(gpu: dict, cpu: dict) -> str:
    lines = []
    if gpu:
        lines.append(f"Latency (batch 1, {gpu.get('device', 'GPU')}): time to first token p50 {gpu.get('ttft_ms_p50', 'n/a')} ms, "
                     f"p90 {gpu.get('ttft_ms_p90', 'n/a')} ms.")
    if cpu:
        lines.append(f"Latency (GGUF on CPU, {cpu.get('device', 'CPU')}): time to first token p50 {cpu.get('ttft_ms_p50')} ms, "
                     f"p90 {cpu.get('ttft_ms_p90')} ms.")
    return "\n\n".join(lines) or "Latency: not measured."


def _table(rows: list[tuple[str, dict]]) -> str:
    out = ["| Model | Exact match | Edit similarity | Parse rate |", "|---|---|---|---|"]
    out += [f"| {n} | {_fmt(x.get('exact_match'))} | {_fmt(x.get('edit_similarity'))} | {_fmt(x.get('parse_rate'))} |" for n, x in rows]
    return "\n".join(out)


def results_section(cfg: dict, m: dict, cpu: dict, he: dict) -> str:
    """The card's Results block for this task."""
    humaneval = (f"pass@1 {he.get('humaneval_pass@1')} (HumanEval+ {he.get('humaneval_plus_pass@1')})" if he.get("status") == "ok"
                 else f"not run ({he.get('reason', 'disabled')})")
    f, b, t, s = m["final"], m["base"], m["teacher"], m.get("sft") or {}
    if cfg["task"] == "fim":
        table = _table([("Teacher", t), ("Student, untrained", b), ("Student, SFT", s), ("**Student, final**", f)])
        return (f"## Results (held-out set, n = {f.get('n')})\n\n{table}\n\n"
                f"{_latency_lines(f.get('latency', {}), cpu)}\n\nHumanEval+: {humaneval}")
    # code: FIM table (with the reference FIM model when comparable) + chat table
    ref = m.get("fim_reference", {})
    fim_rows = [(f"Reference: {cfg['gates'].get('fim_reference_model')}:{ref.get('version')}", ref)] if ref.get("status") == "ok" else []
    fim_rows += [("Student, untrained", b["fim"]), ("Student, SFT", s.get("fim", {})), ("**Student, final**", f["fim"])]
    types = sorted(t["chat"].get("by_type", {}))
    chat_rows = [("Teacher", t["chat"]), ("Student, untrained", b["chat"]), ("Student, SFT", s.get("chat", {})), ("**Student, final**", f["chat"])]
    chat = ["| Model | pass@1 | " + " | ".join(types) + " |", "|---|---|" + "---|" * len(types)]
    chat += [f"| {n} | {_fmt(x.get('pass@1'))} | " + " | ".join(_fmt(x.get("by_type", {}).get(k)) for k in types) + " |"
             for n, x in chat_rows]
    chat_table = "\n".join(chat)
    return (f"## Results: inline completion (held-out FIM spans, n = {f['fim'].get('n')})\n\n{_table(fim_rows)}\n\n"
            f"## Results: chat (held-out problems, tests run in a sandbox, n = {f['chat'].get('n')})\n\n{chat_table}\n\n"
            f"{_latency_lines(f['fim'].get('latency', {}), cpu)}\n\nHumanEval+: {humaneval}")


class _Safe(dict):
    def __missing__(self, key):
        return "n/a"


def _git_commit() -> str:
    try:
        return subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return "not a git checkout"


def _fmt(x) -> str:
    return f"{x:.3f}" if isinstance(x, (int, float)) else "n/a"


def update_catalogue(store: Path, name: str, tag: str, entry: dict, make_live: bool) -> dict:
    path = store / "catalogue.yaml"
    cat = yaml.safe_load(path.read_text()) if path.exists() else {}
    cat = cat or {}
    models = cat.setdefault("models", {})
    m = models.setdefault(name, {"live": None, "versions": {}})
    m["versions"][tag] = entry
    if make_live:
        m["live"] = tag
    path.write_text(yaml.safe_dump(cat, sort_keys=False))
    return cat


def run(run: Run) -> dict:
    cfg = run.cfg
    metrics = read_json(run.step_dir("evaluate") / "metrics.json")
    q = read_json(run.step_dir("quantise") / "summary.json")
    cpu = q.get("cpu_latency") or {}
    g = cfg.get("gates", {})
    if cpu.get("ttft_ms_p50") is not None and "max_ttft_ms_cpu" in g:
        metrics["gates"]["checks"]["ttft_cpu"] = {"value": cpu["ttft_ms_p50"], "threshold": g["max_ttft_ms_cpu"],
                                                  "pass": cpu["ttft_ms_p50"] <= g["max_ttft_ms_cpu"]}
        metrics["gates"]["verdict"] = "pass" if all(c["pass"] for c in metrics["gates"]["checks"].values()) else "fail"
    verdict = metrics["gates"]["verdict"]
    force = cfg.get("register", {}).get("force", False)
    if verdict != "pass" and not force:
        raise RuntimeError("release gates failed; fix and re-run, or set register.force=true to register as non-live")

    prep = read_json(run.step_dir("prepare") / "manifest.json")
    filt = read_json(run.step_dir("verify") / "filter_report.json")
    dest = run.model_store / cfg["name"] / run.tag
    dest.mkdir(parents=True, exist_ok=True)

    artefacts = []
    if "gguf" in q:
        src = run.step_dir("quantise") / q["gguf"]["file"]
        shutil.copy2(src, dest / src.name)
        artefacts.append((src.name, q["gguf"]["bytes"]))
    merged = Path(q["merged"])
    if merged.exists() and not (dest / "merged").exists():
        shutil.copytree(merged, dest / "merged")
    if (dest / "merged").exists():  # list it on re-registration too, not only when first copied
        artefacts.append(("merged/ (fp16, for vLLM)", sum(p.stat().st_size for p in (dest / "merged").rglob("*") if p.is_file())))
    write_json(dest / "metrics.json", metrics)

    lineage = {
        "git_commit": _git_commit(),
        "config_hash": config_hash(cfg),
        "run_dir": str(run.run_dir),
        "sources": prep["sources"],
        "data_sha256": {**prep["sha256"], **filt["sha256"]},
        "gguf_sha256": q.get("gguf", {}).get("sha256"),
    }
    write_json(dest / "lineage.json", lineage)

    he = metrics.get("humaneval", {})
    use = USE.get(cfg["task"], ("", "", ""))
    gate_rows = ["| Check | Value | Threshold | Result |", "|---|---|---|---|"] + [
        f"| {k} | {v.get('value')} | {v.get('threshold', v.get('base', ''))} | {'pass' if v['pass'] else 'FAIL'} |"
        for k, v in metrics["gates"]["checks"].items()
    ]
    data_rows = ["| Repository | Ref | Commit | Licence |", "|---|---|---|---|"] + [
        f"| {repo} | {v['ref']} | `{v['commit'][:10]}` | {v['license']} |" for repo, v in prep["sources"].items()
    ]
    fields = _Safe(
        name=cfg["name"], tag=run.tag, owner=cfg.get("owner", ""), tier=cfg["tier"], task=cfg["task"],
        student=cfg["student"]["model"], teacher=cfg["teacher"]["model"],
        method=cfg.get("method", "SFT on original + verified teacher completions, then offline top-k logit distillation"),
        verdict=verdict.upper() + (" (forced)" if verdict != "pass" else ""),
        registered_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        intended_use=use[0], out_of_scope=use[1], limitations=use[2],
        results_section=results_section(cfg, metrics, cpu, he), serving_notes=SERVING.get(cfg["task"], ""),
        gate_table="\n".join(gate_rows), data_table="\n".join(data_rows),
        decontam=f"{prep['decontam_dropped']}", teacher_filter=f"{filt['teacher_train_filter']}",
        artefact_rows="\n".join(f"| {n} | `{n.split(' ')[0]}` | {sz / 1e6:.0f} MB |" for n, sz in artefacts),
        git_commit=lineage["git_commit"], config_hash=lineage["config_hash"], run_dir=lineage["run_dir"],
        data_hashes=", ".join(f"{k}: `{v[:12]}`" for k, v in lineage["data_sha256"].items()),
    )
    card = (REPO_ROOT / "model_cards" / "template.md").read_text().format_map(fields)
    (dest / "model_card.md").write_text(card)

    entry = {
        "path": str(dest.relative_to(run.model_store)),
        "task": cfg["task"],
        "gguf": q.get("gguf", {}).get("file"),
        "hf_dir": "merged",
        "verdict": verdict,
        "registered_at": fields["registered_at"],
        "config_hash": lineage["config_hash"],
    }
    update_catalogue(run.model_store, cfg["name"], run.tag, entry, make_live=(verdict == "pass"))
    log.info("registered %s:%s at %s (live=%s)", cfg["name"], run.tag, dest, verdict == "pass")
    return {"dest": str(dest), "live": verdict == "pass"}

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
    "fim": ("Inline code completion inside the IDE (fill-in-the-middle), Python first.",
            "Chat, explanations, multi-file refactors (use msci-code-deep), anything outside code.",
            "Trained on public BSD/Apache Python only; no MSCI code yet. Short completions (<= 12 lines). "
            "Inherits teacher mistakes on spans where the teacher was accepted."),
}


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
    verdict = metrics["gates"]["verdict"]
    force = cfg.get("register", {}).get("force", False)
    if verdict != "pass" and not force:
        raise RuntimeError("release gates failed; fix and re-run, or set register.force=true to register as non-live")

    q = read_json(run.step_dir("quantise") / "summary.json")
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

    f, b, t, s = metrics["final"], metrics["base"], metrics["teacher"], metrics.get("sft") or {}
    lat = f.get("latency", {})
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
        method="SFT on original + verified teacher completions, then offline top-k logit distillation",
        verdict=verdict.upper() + (" (forced)" if verdict != "pass" else ""),
        registered_at=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        intended_use=use[0], out_of_scope=use[1], limitations=use[2],
        n_heldout=f.get("n"),
        teacher_em=_fmt(t.get("exact_match")), teacher_es=_fmt(t.get("edit_similarity")), teacher_parse=_fmt(t.get("parse_rate")),
        base_em=_fmt(b.get("exact_match")), base_es=_fmt(b.get("edit_similarity")), base_parse=_fmt(b.get("parse_rate")),
        sft_em=_fmt(s.get("exact_match")), sft_es=_fmt(s.get("edit_similarity")), sft_parse=_fmt(s.get("parse_rate")),
        final_em=_fmt(f.get("exact_match")), final_es=_fmt(f.get("edit_similarity")), final_parse=_fmt(f.get("parse_rate")),
        ttft_p50=lat.get("ttft_ms_p50", "n/a"), ttft_p90=lat.get("ttft_ms_p90", "n/a"), latency_device=lat.get("device", "n/a"),
        humaneval=(f"pass@1 {he.get('humaneval_pass@1')} (HumanEval+ {he.get('humaneval_plus_pass@1')})" if he.get("status") == "ok"
                   else f"not run ({he.get('reason', 'disabled')})"),
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

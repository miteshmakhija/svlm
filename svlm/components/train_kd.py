"""Step 6 — train_kd: continue from the SFT adapter with offline top-k logit distillation.

loss = alpha * CE + (1 - alpha) * T^2 * KL(teacher_topk || student_topk)
Output: runs/.../train_kd/adapter/
"""
from __future__ import annotations

from ..config import Run
from ..data import load_topk_shards
from ..modeling import free_gpu, load_student_for_training
from ..training import train
from ..utils import log, write_json

STEP = "train_kd"


def run(run: Run) -> dict:
    cfg = run.cfg
    if not cfg["train"].get("kd", {}).get("enabled", True):
        log.info("%s skipped: train.kd.enabled=false", STEP)
        return {"skipped": "train.kd.enabled=false"}
    out = run.step_dir(STEP)
    sft_adapter = run.step_dir("train_sft") / "adapter"
    if not sft_adapter.exists():
        raise FileNotFoundError(f"{sft_adapter} missing; run train_sft first")
    rows = load_topk_shards(run.step_dir("teacher_logits"))
    if not rows:
        raise FileNotFoundError("no teacher top-k shards; run teacher_logits first")
    feats = []
    for r in rows:
        ids, s = r["input_ids"], r["target_start"]
        feats.append({**r, "labels": [-100] * s + ids[s:]})
    log.info("train_kd: %d examples with teacher top-k", len(feats))

    model, tok = load_student_for_training(cfg, adapter_dir=sft_adapter)
    k = cfg["teacher"]["logits"]["top_k"]
    kd = cfg["train"]["kd"]
    try:
        adapter = train(model, tok, feats, out, kd, cfg.get("seed", 42), kd={"alpha": kd["alpha"], "temperature": kd["temperature"]}, k=k)
    finally:
        model = None
        free_gpu()
    info = {"adapter": str(adapter), "n_examples": len(feats), "alpha": kd["alpha"], "temperature": kd["temperature"]}
    write_json(out / "summary.json", info)
    return info

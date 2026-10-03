"""Step 5 — train_sft: LoRA supervised fine-tuning on original + teacher-accepted targets.

Loss is computed on the completion tokens only (the FIM prompt is masked).
Output: runs/.../train_sft/adapter/  (checkpoints/ kept for resume)
"""
from __future__ import annotations

from ..config import Run
from ..data import encode_example
from ..modeling import free_gpu, load_student_for_training
from ..training import train
from ..utils import log, read_jsonl, write_json

STEP = "train_sft"


def run(run: Run) -> dict:
    cfg = run.cfg
    out = run.step_dir(STEP)
    model, tok = load_student_for_training(cfg)
    rows = read_jsonl(run.step_dir("verify") / "sft.jsonl")
    feats = [f for f in (encode_example(tok, r, cfg["task"], cfg["student"]["max_seq_len"]) for r in rows) if f]
    log.info("train_sft: %d/%d examples fit max_seq_len", len(feats), len(rows))
    try:
        adapter = train(model, tok, feats, out, cfg["train"]["sft"], cfg.get("seed", 42))
    finally:
        free_gpu(model)
    info = {"adapter": str(adapter), "n_examples": len(feats)}
    write_json(out / "summary.json", info)
    return info

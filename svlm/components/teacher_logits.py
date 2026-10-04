"""Step 4 — teacher_logits (offline logit distillation data).

The teacher re-reads every SFT example (teacher forcing) and we keep its top-k next-token
distribution at each target position. Teacher and student must share a tokenizer; this is
checked before any compute is spent.

Storage: shard_XXXXX.npz with flat arrays + offsets (see svlm.data.save_topk_shard).
Row j of topk_* for an example is the teacher's distribution for target token j, taken from the
logits one position earlier (causal shift).
"""
from __future__ import annotations

import numpy as np
import torch

from ..config import Run
from ..data import encode_example, save_topk_shard
from ..modeling import free_gpu, load_causal_lm, load_tokenizer
from ..utils import chunks, log, read_jsonl, write_json

STEP = "teacher_logits"
SHARD = 200


def check_same_tokenizer(teacher_name: str, student_name: str) -> None:
    t, s = load_tokenizer(teacher_name), load_tokenizer(student_name)
    probe = "def f(x):\n    return x ** 2  # <|fim_prefix|> ünïcødé"
    if t(probe)["input_ids"] != s(probe)["input_ids"] or len(t) != len(s):
        raise ValueError(f"teacher {teacher_name} and student {student_name} do not share a tokenizer; "
                         "logit distillation needs one (use SFT only, or pick a matching pair)")


@torch.no_grad()
def topk_for_batch(model, feats: list[dict], pad_id: int, k: int) -> list[tuple[np.ndarray, np.ndarray]]:
    L = max(len(f["input_ids"]) for f in feats)
    ids = torch.full((len(feats), L), pad_id, dtype=torch.long)
    mask = torch.zeros_like(ids)
    for i, f in enumerate(feats):
        ids[i, : len(f["input_ids"])] = torch.tensor(f["input_ids"])
        mask[i, : len(f["input_ids"])] = 1
    logits = model(input_ids=ids.to(model.device), attention_mask=mask.to(model.device)).logits
    out = []
    for i, f in enumerate(feats):
        s, n = f["target_start"], len(f["input_ids"])
        lg = logits[i, s - 1 : n - 1].float()               # predicts tokens s..n-1
        lp = torch.log_softmax(lg, dim=-1)
        v, ix = lp.topk(k, dim=-1)
        out.append((ix.cpu().numpy().astype(np.int32), v.cpu().numpy().astype(np.float16)))
    del logits
    return out


def run(run: Run) -> dict:
    cfg = run.cfg
    t = cfg["teacher"]
    k = t["logits"]["top_k"]
    out = run.step_dir(STEP)
    check_same_tokenizer(t["model"], cfg["student"]["model"])

    tok = load_tokenizer(cfg["student"]["model"], padding_side="right")
    rows = read_jsonl(run.step_dir("verify") / "sft.jsonl")
    feats = [f for f in (encode_example(tok, r, cfg["task"], cfg["student"]["max_seq_len"]) for r in rows) if f]
    log.info("teacher_logits: %d/%d examples fit max_seq_len", len(feats), len(rows))

    model = None
    n_tokens = 0
    try:
        for si, shard in enumerate(chunks(feats, SHARD)):
            path = out / f"shard_{si:05d}.npz"
            if path.exists():
                continue
            if model is None:
                model = load_causal_lm(t["model"], load_in_4bit=t.get("load_in_4bit", True))
            order = sorted(range(len(shard)), key=lambda i: len(shard[i]["input_ids"]))
            results: list = [None] * len(shard)
            for b in chunks(order, t["logits"].get("batch_size", 4)):
                for i, res in zip(b, topk_for_batch(model, [shard[i] for i in b], tok.pad_token_id, k)):
                    results[i] = res
            save_topk_shard(
                path,
                [f["id"] for f in shard],
                [f["target_start"] for f in shard],
                [f["input_ids"] for f in shard],
                [r[0] for r in results],
                [r[1] for r in results],
            )
            n_tokens += sum(len(r[0]) for r in results)
            log.info("teacher_logits shard %d done", si + 1)
    finally:
        model = None
        free_gpu()
    info = {"n_examples": len(feats), "top_k": k, "new_target_tokens": n_tokens}
    write_json(out / "summary.json", info)
    return info

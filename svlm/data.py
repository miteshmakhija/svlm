"""Tokenisation and batching shared by teacher_logits, train_sft, train_kd and evaluate.

Teacher and student share one tokenizer, and every step uses `encode_example`, so token ids,
target positions and the teacher's stored top-k rows always line up.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .chat import split_for_training
from .fim import EOS, fim_prompt


def prompt_and_target(row: dict) -> tuple[str, str]:
    """Training text for one SFT row. Rows are FIM spans (prefix/suffix/target, the default) or chat
    conversations (kind="chat", messages ending with the assistant reply)."""
    if row.get("kind") == "chat":
        return split_for_training(row["messages"])
    return fim_prompt(row["prefix"], row["suffix"]), row["target"] + EOS


def encode_example(tok, row: dict, task: str, max_len: int) -> dict | None:
    """input_ids = prompt + target; labels mask the prompt, so loss is on the completion / reply only.
    None if too long. `task` is kept for the call sites; the row itself says how to format it."""
    prompt, target = prompt_and_target(row)
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    t = tok(target, add_special_tokens=False)["input_ids"]
    if len(p) + len(t) > max_len or not t:
        return None
    return {"input_ids": p + t, "labels": [-100] * len(p) + t, "target_start": len(p), "id": row["id"]}


class PadCollator:
    """Right-pads input_ids / labels; carries optional teacher top-k tensors."""

    def __init__(self, pad_id: int, k: int | None = None):
        self.pad_id, self.k = pad_id, k

    def __call__(self, feats: list[dict]) -> dict:
        L = max(len(f["input_ids"]) for f in feats)
        ids = torch.full((len(feats), L), self.pad_id, dtype=torch.long)
        labels = torch.full((len(feats), L), -100, dtype=torch.long)
        mask = torch.zeros((len(feats), L), dtype=torch.long)
        for i, f in enumerate(feats):
            n = len(f["input_ids"])
            ids[i, :n] = torch.tensor(f["input_ids"])
            labels[i, :n] = torch.tensor(f["labels"])
            mask[i, :n] = 1
        batch = {"input_ids": ids, "attention_mask": mask, "labels": labels}
        if self.k and "topk_ids" in feats[0]:
            # teacher rows aligned to label positions; zero rows where there is no target
            tids = torch.zeros((len(feats), L, self.k), dtype=torch.long)
            tlp = torch.full((len(feats), L, self.k), -1e4, dtype=torch.float32)
            for i, f in enumerate(feats):
                s, m = f["target_start"], len(f["topk_ids"])
                tids[i, s : s + m] = torch.as_tensor(f["topk_ids"], dtype=torch.long)
                tlp[i, s : s + m] = torch.as_tensor(f["topk_logprobs"], dtype=torch.float32)
            batch["topk_ids"], batch["topk_logprobs"] = tids, tlp
        return batch


# ---------- teacher top-k storage ----------

def save_topk_shard(path: Path, ids: list[str], target_starts: list[int], input_ids: list[list[int]],
                    topk_ids: list[np.ndarray], topk_lp: list[np.ndarray]) -> None:
    lens = np.array([len(x) for x in input_ids], dtype=np.int32)
    tlen = np.array([len(x) for x in topk_ids], dtype=np.int32)
    np.savez_compressed(
        path,
        ids=np.array(ids),
        target_start=np.array(target_starts, dtype=np.int32),
        input_lens=lens,
        input_ids=np.concatenate([np.array(x, dtype=np.int32) for x in input_ids]),
        topk_lens=tlen,
        topk_ids=np.concatenate(topk_ids).astype(np.int32),
        topk_lp=np.concatenate(topk_lp).astype(np.float16),
    )


def load_topk_shards(d: Path) -> list[dict]:
    rows: list[dict] = []
    for p in sorted(d.glob("shard_*.npz")):
        z = np.load(p)
        io = np.concatenate([[0], np.cumsum(z["input_lens"])])
        to = np.concatenate([[0], np.cumsum(z["topk_lens"])])
        for i, rid in enumerate(z["ids"]):
            rows.append(
                {
                    "id": str(rid),
                    "target_start": int(z["target_start"][i]),
                    "input_ids": z["input_ids"][io[i] : io[i + 1]].tolist(),
                    "topk_ids": z["topk_ids"][to[i] : to[i + 1]],
                    "topk_logprobs": z["topk_lp"][to[i] : to[i + 1]].astype(np.float32),
                }
            )
    return rows

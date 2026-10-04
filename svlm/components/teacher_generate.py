"""Step 2 — teacher_generate.

The teacher fills the hole for:
  * every held-out sample  -> the teacher's score is the quality ceiling used by the gates
  * a share of train samples (teacher_augment_fraction) -> extra, teacher-written targets

Backends: "hf" (transformers + bitsandbytes 4-bit; default, fewest dependencies) or "vllm"
(faster; install per notebook 00). Output is sharded and resumable.

Outputs: runs/.../teacher_generate/{heldout,train}/shard_*.jsonl
"""
from __future__ import annotations

import random

from .. import tasks
from ..config import Run
from ..fim import STOP_STRINGS, fim_prompt, trim_completion
from ..modeling import free_gpu, generate_batch, load_causal_lm, load_tokenizer
from ..utils import chunks, log, read_jsonl, run_sharded, write_json

STEP = "teacher_generate"


class _HFGenerator:
    def __init__(self, cfg: dict):
        t = cfg["teacher"]
        self.tok = load_tokenizer(t["model"], padding_side="left")
        self.model = load_causal_lm(t["model"], load_in_4bit=t.get("load_in_4bit", True))
        self.gen = t["gen"]

    def __call__(self, prompts: list[str]) -> list[str]:
        out: list[str] = []
        for batch in chunks(prompts, self.gen.get("batch_size", 16)):
            out.extend(generate_batch(self.model, self.tok, batch, self.gen["max_new_tokens"], self.gen.get("temperature", 0.0)))
        return out

    def close(self):
        self.model = None
        free_gpu()


class _VLLMGenerator:
    def __init__(self, cfg: dict):
        from vllm import LLM, SamplingParams

        t = cfg["teacher"]
        kwargs = dict(model=t["model"], dtype="half", max_model_len=cfg["student"].get("max_seq_len", 1024) + t["gen"]["max_new_tokens"],
                      gpu_memory_utilization=t.get("gpu_memory_utilization", 0.85), seed=cfg.get("seed", 42))
        if t.get("load_in_4bit", True) and "awq" not in t["model"].lower():
            kwargs["quantization"] = "bitsandbytes"
        self.llm = LLM(**kwargs)
        self.params = SamplingParams(temperature=t["gen"].get("temperature", 0.0), max_tokens=t["gen"]["max_new_tokens"], stop=STOP_STRINGS)

    def __call__(self, prompts: list[str]) -> list[str]:
        res = self.llm.generate(prompts, self.params, use_tqdm=False)
        return [r.outputs[0].text for r in res]

    def close(self):
        self.llm = None
        free_gpu()


def make_generator(cfg: dict):
    backend = cfg["teacher"].get("backend", "hf")
    return _VLLMGenerator(cfg) if backend == "vllm" else _HFGenerator(cfg)


def run(run: Run) -> dict:
    cfg = run.cfg
    if cfg["task"] != "fim":
        return tasks.get(cfg["task"]).teacher_generate(run)
    prep = run.step_dir("prepare")
    out = run.step_dir(STEP)
    held = read_jsonl(prep / "heldout.jsonl")
    train = read_jsonl(prep / "train.jsonl")
    frac = cfg["data"].get("teacher_augment_fraction", 0.0)
    rng = random.Random(cfg.get("seed", 42))
    aug = rng.sample(train, int(len(train) * frac)) if frac > 0 else []

    gen = make_generator(cfg)

    def work(rows: list[dict]) -> list[dict]:
        # sort by length inside the shard so padded batches waste less compute
        order = sorted(range(len(rows)), key=lambda i: len(rows[i]["prefix"]) + len(rows[i]["suffix"]))
        prompts = [fim_prompt(rows[i]["prefix"], rows[i]["suffix"]) for i in order]
        raw = gen(prompts)
        res: list[dict | None] = [None] * len(rows)
        for j, i in enumerate(order):
            r = rows[i]
            res[i] = {
                "id": r["id"],
                "split": r["split"],
                "teacher_raw": raw[j],
                "teacher_completion": trim_completion(raw[j], r["span_type"], r["suffix"]),
            }
        return res  # type: ignore[return-value]

    try:
        h = run_sharded(held, out / "heldout", work, shard_size=200, desc="teacher heldout")
        a = run_sharded(aug, out / "train", work, shard_size=200, desc="teacher train") if aug else []
    finally:
        gen.close()
    info = {"teacher": cfg["teacher"]["model"], "backend": cfg["teacher"].get("backend", "hf"), "n_heldout": len(h), "n_train": len(a)}
    write_json(out / "summary.json", info)
    log.info("teacher_generate: %s", info)
    return info

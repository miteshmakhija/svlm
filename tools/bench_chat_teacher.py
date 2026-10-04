"""Measure the chat teacher (vLLM or hf) on real MBPP problems and estimate the full teacher step.

Run as its own process (the notebook does), so GPU memory is released on exit. Prints generated
tokens/s, the share of answers that pass MBPP's tests, and the projected time for teacher_generate
with the sizes in the config. The last stdout line is a JSON summary.

    python tools/bench_chat_teacher.py --config configs/code.yaml --problems 48
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svlm import sandbox  # noqa: E402
from svlm.chat import extract_code  # noqa: E402
from svlm.config import load_config, parse_overrides  # noqa: E402
from svlm.tasks.code import SOLVE_SYSTEM, _mbpp, solve_prompt  # noqa: E402
from svlm.teacher import ChatTeacher  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/code.yaml")
    ap.add_argument("--problems", type=int, default=48)
    ap.add_argument("--set", action="append", default=[])
    a = ap.parse_args()
    cfg = load_config(a.config, parse_overrides(a.set))
    g, c = cfg["teacher"]["gen"], cfg["data"]["code"]
    n = g.get("n_solutions", 4)
    probs = _mbpp("validation")[: a.problems]

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg["teacher"]["model"])
    t0 = time.time()
    teacher = ChatTeacher(cfg)
    load_s = time.time() - t0
    convs = [[{"role": "system", "content": SOLVE_SYSTEM}, {"role": "user", "content": solve_prompt(p["task"], p["tests"])}] for p in probs]
    t0 = time.time()
    replies = teacher.chat(convs, n=n, temperature=g.get("temperature", 0.7), max_tokens=g.get("max_new_tokens", 768))
    dt = time.time() - t0
    teacher.close()
    toks = [len(tok(r, add_special_tokens=False)["input_ids"]) for rs in replies for r in rs]
    res = sandbox.run_many([(extract_code(rs[0]), p["tests"], p["setup"]) for p, rs in zip(probs, replies)])
    tps = sum(toks) / dt
    avg = sum(toks) / len(toks)

    # projected teacher_generate: problems (1 x ~450 tokens), solutions (n x avg), held-out (1 x avg), explain (1 x ~200)
    n_mbpp = min(384, c.get("mbpp_max") or 384) if c.get("mbpp", True) else 0
    seeds = c["synth_train"] + c["synth_heldout"]
    tokens = seeds * 450 + (n_mbpp + c["synth_train"] * 0.8) * n * avg + (90 + c["synth_heldout"]) * avg + c["explain"] * 200
    hours = tokens / tps / 3600
    print(f"loaded {cfg['teacher']['model']} ({cfg['teacher'].get('backend', 'vllm')}) in {load_s:.0f}s")
    print(f"{len(probs)} problems x {n} answers: {sum(toks)} tokens in {dt:.0f}s -> {tps:.0f} tokens/s, {avg:.0f} tokens/answer")
    print(f"first answer passes MBPP tests: {sum(r.passed for r in res)}/{len(res)}")
    print(f"projected teacher_generate for this config: ~{tokens / 1e6:.1f}M tokens, ~{hours:.1f} h (+ ~{load_s / 60:.0f} min load per stage restart)")
    print(json.dumps({"tokens_per_s": round(tps), "tokens_per_answer": round(avg), "pass_first": sum(r.passed for r in res) / len(res),
                      "projected_hours": round(hours, 2), "load_s": round(load_s)}))


if __name__ == "__main__":
    main()

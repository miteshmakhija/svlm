"""Load the teacher in 4-bit and measure batched FIM generation throughput.

Run as its own process (notebook 00 does this) so the GPU memory is fully released on exit;
holding the teacher in the notebook kernel leaves ~5 GB that the pipeline then cannot use.
The last stdout line is a JSON summary.

    python tools/bench_teacher.py --config configs/code_fast.yaml
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch  # noqa: E402
import yaml  # noqa: E402

from svlm.fim import fim_prompt  # noqa: E402
from svlm.modeling import generate_batch, load_causal_lm, load_tokenizer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/code_fast.yaml")
    ap.add_argument("--batch", type=int, default=16)
    a = ap.parse_args()

    cfg = yaml.safe_load(open(a.config))
    name = cfg["teacher"]["model"]
    torch.cuda.reset_peak_memory_stats()
    t0 = time.time()
    tok = load_tokenizer(name)
    teacher = load_causal_lm(name, load_in_4bit=True)
    load_s = time.time() - t0
    weights_gb = torch.cuda.memory_allocated() / 1e9
    print(f"loaded {name} in {load_s:.0f}s, weights use {weights_gb:.1f} GB")

    prefix = "import pandas as pd\n\ndef monthly_returns(prices: pd.Series) -> pd.Series:\n    \"\"\"Month-end simple returns.\"\"\"\n"
    suffix = "\n    return rets.dropna()\n"
    prompts = [fim_prompt(prefix, suffix)] * a.batch
    t0 = time.time()
    outs = generate_batch(teacher, tok, prompts, max_new_tokens=128)
    dt = time.time() - t0
    n_new = sum(len(tok(o, add_special_tokens=False)["input_ids"]) for o in outs)
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"batch of {a.batch}: {n_new} new tokens in {dt:.1f}s -> {n_new/dt:.0f} tokens/s | peak memory {peak_gb:.1f} GB")
    print("--- sample completion ---")
    print(outs[0].split("<|endoftext|>")[0])
    print(json.dumps({"model": name, "load_s": round(load_s), "weights_gb": round(weights_gb, 1),
                      "tokens_per_s": n_new / dt, "peak_gb": round(peak_gb, 1)}))


if __name__ == "__main__":
    main()

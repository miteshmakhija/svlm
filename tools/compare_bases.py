"""Which checkpoint should msci-code start from? Score untrained candidates on both jobs.

FIM: held-out spans from an existing prepare step (default: msci-code-fast's run, the spans msci-code
is gated against). Chat: MBPP validation problems, tests run in the sandbox. Uses the pipeline's own
evaluation code, so the numbers match what `evaluate` will report for the untrained base.

    python tools/compare_bases.py --config configs/code.yaml \
        --models Qwen/Qwen2.5-Coder-1.5B Qwen/Qwen2.5-Coder-1.5B-Instruct --n-fim 150
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svlm.components.evaluate import eval_variant  # noqa: E402
from svlm.config import load_config  # noqa: E402
from svlm.tasks.code import _mbpp, chat_variant  # noqa: E402
from svlm.utils import iter_jsonl, read_jsonl  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/code.yaml")
    ap.add_argument("--models", nargs="+", default=["Qwen/Qwen2.5-Coder-1.5B", "Qwen/Qwen2.5-Coder-1.5B-Instruct"])
    ap.add_argument("--prepare-dir", help="a prepare step folder with heldout.jsonl + files.jsonl "
                                          "(default: $SVLM_ROOT/runs/msci-code-fast/0.1.0-t4/prepare)")
    ap.add_argument("--n-fim", type=int, default=150)
    ap.add_argument("--n-chat", type=int, default=90)
    a = ap.parse_args()

    cfg = load_config(a.config)
    root = Path(cfg["paths"]["root"])
    prep = Path(a.prepare_dir) if a.prepare_dir else root / "runs" / "msci-code-fast" / "0.1.0-t4" / "prepare"
    held = read_jsonl(prep / "heldout.jsonl")[: a.n_fim]
    files = {f["file_id"]: f["content"] for f in iter_jsonl(prep / "files.jsonl")}
    problems = _mbpp("validation")[: a.n_chat]
    out_root = root / "runs" / "compare_bases"
    rows = []
    for m in a.models:
        c = {**cfg, "student": {**cfg["student"], "model": m}}
        out = out_root / m.replace("/", "__")
        out.mkdir(parents=True, exist_ok=True)
        fim = eval_variant(c, None, held, files, out, "base")
        chat = chat_variant(c, None, problems, out, "base")
        rows.append({"model": m, "fim_edit_similarity": fim["edit_similarity"], "fim_exact_match": fim["exact_match"],
                     "fim_parse_rate": fim["parse_rate"], "chat_pass@1": chat["pass@1"]})
    print("\n| Model | FIM edit sim | FIM exact | FIM parse | Chat pass@1 (MBPP val) |\n|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['model']} | {r['fim_edit_similarity']:.3f} | {r['fim_exact_match']:.3f} | {r['fim_parse_rate']:.3f} | {r['chat_pass@1']:.3f} |")
    (out_root / "summary.json").write_text(json.dumps(rows, indent=2))
    print("\nTraining fixes what the mix teaches: FIM comes back easily (2,500 original spans), chat ability is harder to")
    print("create from nothing. Prefer the checkpoint with the better chat score unless its FIM score is far below.")
    if os.environ.get("SVLM_ROOT") is None:
        print("(SVLM_ROOT not set: using paths.root from the config)")


if __name__ == "__main__":
    main()

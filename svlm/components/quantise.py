"""Step 8 — quantise: merge the final adapter and export serving formats.

* merged/            fp16 Hugging Face model (vLLM on GPU can serve this directly)
* model.<type>.gguf  for llama.cpp on CPU (the gateway's default backend)
      q8_0 / f16 / bf16 come straight from llama.cpp's converter (no compile step);
      q4_k_m additionally builds llama-quantize (~5 min on Colab).
* awq/ (optional)    4-bit AWQ via llm-compressor, for larger models on GPU
"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..config import Run
from ..modeling import merge_and_save
from ..utils import log, sha256_file, write_json

STEP = "quantise"
DIRECT_TYPES = {"f16", "bf16", "q8_0", "f32"}


def _llama_cpp(cache: Path) -> tuple[Path, str]:
    repo = cache / "llama.cpp"
    if not (repo / ".git").exists():
        subprocess.run(["git", "clone", "--depth", "1", "https://github.com/ggml-org/llama.cpp", str(repo)], check=True)
    sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    return repo, sha


def to_gguf(merged: Path, out: Path, outtype: str, cache: Path) -> dict:
    repo, sha = _llama_cpp(cache)
    outtype = outtype.lower()
    first = outtype if outtype in DIRECT_TYPES else "f16"
    gguf = out / f"model.{first}.gguf"
    if not gguf.exists():
        subprocess.run([sys.executable, str(repo / "convert_hf_to_gguf.py"), str(merged), "--outfile", str(gguf), "--outtype", first],
                       check=True, env={**os.environ})
    final = gguf
    if outtype not in DIRECT_TYPES:
        qbin = repo / "build" / "bin" / "llama-quantize"
        if not qbin.exists():
            subprocess.run(["cmake", "-B", "build", "-DGGML_CUDA=OFF", "-DLLAMA_CURL=OFF"], cwd=repo, check=True)
            subprocess.run(["cmake", "--build", "build", "--target", "llama-quantize", "-j", str(os.cpu_count() or 2)], cwd=repo, check=True)
        final = out / f"model.{outtype}.gguf"
        if not final.exists():
            subprocess.run([str(qbin), str(gguf), str(final), outtype.upper()], check=True)
    return {"file": final.name, "type": outtype, "bytes": final.stat().st_size, "sha256": sha256_file(final), "llama_cpp_commit": sha}


def to_awq(merged: Path, out: Path) -> dict:
    from llmcompressor import oneshot
    from llmcompressor.modifiers.awq import AWQModifier

    target = out / "awq"
    if not (target / "config.json").exists():
        oneshot(model=str(merged), dataset="open_platypus", recipe=[AWQModifier(scheme="W4A16", targets=["Linear"], ignore=["lm_head"])],
                output_dir=str(target), max_seq_length=1024, num_calibration_samples=128)
    return {"dir": "awq", "scheme": "W4A16"}


def run(run: Run) -> dict:
    cfg = run.cfg
    q = cfg.get("quantise", {})
    out = run.step_dir(STEP)
    metrics_path = run.step_dir("evaluate") / "metrics.json"
    from ..utils import read_json

    adapter = Path(read_json(metrics_path)["final_adapter"]) if metrics_path.exists() else run.step_dir("train_kd") / "adapter"
    merged = merge_and_save(cfg, adapter, run.run_dir / "merged")
    info: dict = {"merged": str(merged)}
    if q.get("gguf_outtype"):
        info["gguf"] = to_gguf(merged, out, q["gguf_outtype"], run.root / "cache")
    if q.get("awq"):
        info["awq"] = to_awq(merged, out)
    write_json(out / "summary.json", info)
    log.info("quantise: %s", info)
    return info

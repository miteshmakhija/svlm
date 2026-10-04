# SVLM Model Catalogue

Distillation pipeline for MSCI's small vertical language models: a large open-weight **teacher**
trains a small **student**, which is then evaluated, quantised and registered in a model store
that the catalogue gateway serves to VS Code and JetBrains IDEs.

| Model | Job | Student | Teacher | Status |
|---|---|---|---|---|
| `msci-code-fast` | Inline completion (fill-in-the-middle) | Qwen2.5-Coder-0.5B | Qwen2.5-Coder-7B (4-bit) | **M1, live** (`0.1.0-t4`) |
| `msci-code` | One coder for inline completion **and** chat (replaces the planned `msci-code-deep`) | Qwen2.5-Coder-1.5B(-Instruct) | Qwen2.5-Coder-14B-Instruct-AWQ (vLLM) | **built, notebook 02** |
| `msci-general` | Summaries, Excel formulas, slides | Qwen3-1.7B + 3 LoRA adapters | Qwen3-14B-AWQ | next |
| `msci-reason` | Multi-step reasoning | Qwen3-1.7B | Qwen3-14B-AWQ (thinking) | after general |

Design document: *SVLM Model Catalogue* (claude.ai artifact, v0.2).

## How it works

```
prepare -> teacher_generate -> verify -> teacher_logits -> train_sft -> train_kd -> evaluate -> quantise -> register
```

| Step | Does | Main output |
|---|---|---|
| `prepare` | Clone pinned repos, cut FIM spans, remove benchmark overlap (13-gram), split by file | `train.jsonl`, `heldout.jsonl`, `manifest.json` |
| `teacher_generate` | Teacher fills every held-out hole (quality ceiling) and 25% of train holes | sharded JSONL |
| `verify` | Keep teacher completions that parse in the full file and stay close to the original | `sft.jsonl`, `filter_report.json` |
| `teacher_logits` | Teacher's top-20 next-token log-probs on every target token | `shard_*.npz` |
| `train_sft` | LoRA SFT, loss on completion tokens only | `adapter/` |
| `train_kd` | Continue with `0.5·CE + 0.5·KL(teacher_top20 ‖ student)` | `adapter/` |
| `evaluate` | Base vs SFT vs final vs teacher; latency; HumanEval+; release gates | `metrics.json` |
| `quantise` | Merge LoRA; export GGUF (CPU) and fp16 HF dir (GPU) | `model.q8_0.gguf`, `merged/` |
| `register` | Copy to the model store, write model card, update `catalogue.yaml` | `model_store/...` |

Every step writes a `_SUCCESS` marker and long steps save shards/checkpoints to Google Drive, so a
Colab disconnect costs minutes, not hours. Re-run the same command to continue.

## Quick start on Colab (T4)

1. Copy this folder to Google Drive as `MyDrive/svlm/svlm-catalogue` (Drive for Desktop is easiest).
2. Open `notebooks/00_smoke_test.ipynb` in Colab with a T4 runtime and run all cells (~30–45 min).
   It checks the environment, runs the whole pipeline on tiny data with the real models, estimates
   the full run's hours and compute units, and writes a version lock file.
3. Open `notebooks/01_code_fast.ipynb` and run the steps. Plan sessions using the estimate from step 2.
4. The released model lands in `MyDrive/svlm/model_store/msci-code-fast/0.1.0-t4/`.
5. `notebooks/02_code.ipynb` (L4 runtime) builds `msci-code`: vLLM teacher, sandbox-verified chat data
   mixed with FIM, and gates that compare its FIM quality with the live `msci-code-fast` on the same spans.

Command line equivalent (any machine with a GPU):

```bash
export SVLM_ROOT=/path/to/storage
python pipelines/run_local.py --config configs/code_fast.yaml                 # all steps
python pipelines/run_local.py --config configs/code_fast.yaml --status        # what's done
python pipelines/run_local.py --config configs/code_fast.yaml --from train_sft
python pipelines/run_local.py --config configs/code_fast.yaml --set teacher.model=Qwen/Qwen2.5-Coder-14B
```

## Repository layout

```
configs/            one YAML per catalogue model
svlm/
  config.py         config loading, run-directory layout, step markers
  fim.py            FIM span construction, prompt format, trimming, scoring
  decontam.py       13-gram benchmark decontamination
  data.py           tokenisation, padding collator, teacher top-k storage
  kd_loss.py        offline top-k KL + CE loss
  modeling.py       T4-safe model loading (fp16/NF4), LoRA, generation, merging
  training.py       Trainer setup for SFT and KD, checkpoint resume
  chat.py           ChatML formatting shared by training, evaluation and the gateway; code extraction
  sandbox.py        runs model-written code against asserts in a short-lived, resource-limited process
  teacher.py        chat teacher with vLLM (default) or transformers backends
  components/       the nine pipeline steps; `task: fim` is handled here, other tasks by svlm/tasks/
  tasks/code.py     task `code` (msci-code): prepare, teacher_generate, verify, evaluate + gates
pipelines/
  run_local.py      runs steps in order (Colab, workstation, or one Kubeflow pod per step)
  kfp_pipeline.py   Kubeflow Pipelines v2 definition (cluster tier)
gateway/            FastAPI gateway serving the model store's GGUFs on CPU (llama.cpp), Continue profile
tools/              notebook generator, teacher benchmarks, base-checkpoint comparison
notebooks/          00_smoke_test, 01_code_fast, 02_code (generated by tools/make_notebooks.py)
model_cards/        model card template
requirements/       colab.txt, vllm.txt (optional), dev.txt
tests/              unit tests + CPU end-to-end dry run with tiny random models
```

## Local development (CPU, no GPU needed)

```bash
pip install -r requirements/dev.txt
pytest -q                          # unit tests (~5 s)
python tests/e2e_tiny.py           # all nine steps with tiny random models (~3 min)
python tests/e2e_tiny.py --task code   # same for msci-code: MBPP download, hf teacher, sandbox (~2 min)
python pipelines/kfp_pipeline.py --config configs/code_fast.yaml --image <registry>/svlm:0.1.0 --out code_fast.yaml
```

The e2e run checks wiring, file formats, resume markers and the model card, not model quality.

## T4 notes

* No bf16 on T4: everything runs fp16 (or bf16 automatically on L4/A100).
* Teacher loads in 4-bit NF4 via bitsandbytes (~5 GB for 7B, ~9.5 GB for 14B).
* Teacher and student never share the GPU: each step loads one model and frees it.
* Out of memory: lower `teacher.gen.batch_size`, `teacher.logits.batch_size` or `train.*.per_device_batch`
  (raise `grad_accum` to keep the effective batch).
* vLLM is optional for this model. It installs its own torch, so use it in a separate runtime (notebook 00).

## Data and licences

Training data is public code at pinned tags (pandas, NumPy, scikit-learn: BSD-3-Clause; requests:
Apache-2.0). Commit hashes and licences are recorded in `manifest.json` and the model card. No MSCI code
or data is used on Colab; internal data belongs to the cluster tier only.

## Next milestones

* **msci-code** full run on Colab (notebook 02), then `msci-general` + router, then `msci-reason`
* Serving speed: on a 4-core laptop CPU, llama.cpp prefills ~190 tokens/s with the 0.5B model, so a full
  ~600-token completion prompt takes ~3–4 s before the first token (1.5B: ~3x longer). Shorter IDE prompts,
  a bigger CPU server or one small GPU (design section 12) are the levers.
* **M5** gateway (FastAPI/Uvicorn, router, per-user keys, audit log) + Continue profiles
* **M6** teammate pilot · **M7** Tech Fest demo

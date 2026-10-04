"""Generate the Colab notebooks from this file, so they stay reviewable as plain Python.

    python tools/make_notebooks.py
"""
from pathlib import Path

import nbformat as nbf

OUT = Path(__file__).resolve().parents[1] / "notebooks"

SETUP = r'''# --- Setup: GPU check, Google Drive, repo, packages (run every session) ---
import glob, json, os, subprocess, sys
from google.colab import drive

drive.mount('/content/drive')
SVLM_ROOT = '/content/drive/MyDrive/svlm'       # run outputs, checkpoints and the model store live here
REPO_ON_DRIVE = f'{SVLM_ROOT}/svlm-catalogue'   # copy the repo folder here once (Drive for Desktop works)
REPO = '/content/svlm-catalogue'                # working copy on Colab's fast local disk
os.environ['SVLM_ROOT'] = SVLM_ROOT
os.makedirs(SVLM_ROOT, exist_ok=True)

def _find_repo():
    """Return the first folder that looks like the repo (has configs/ and svlm/), searching
    Colab's local disk and Drive, including one extra nesting level from zip/folder uploads."""
    # Drive first: re-running this cell then refreshes the local copy with whatever was synced to Drive
    bases = [REPO_ON_DRIVE, '/content/drive/MyDrive/svlm-catalogue', '/content/svlm-catalogue', '/content/svlm', '/content']
    for b in bases:
        for cand in (b, os.path.join(b, 'svlm-catalogue')):
            if os.path.isdir(os.path.join(cand, 'configs')) and os.path.isdir(os.path.join(cand, 'svlm')):
                return cand
    return None

found = _find_repo()
if found is None:
    from google.colab import files
    print('Repo not found. Upload svlm-catalogue.zip (from C:\\projects\\TechFest):')
    up = files.upload()
    subprocess.run(['unzip', '-q', '-o', next(iter(up)), '-d', '/content'], check=True)
    found = _find_repo()
    assert found, 'zip did not contain the svlm-catalogue folder'
if found.startswith('/content/drive'):
    # work from a copy on Colab's local disk: faster, and Drive stays the clean source
    subprocess.run(['rm', '-rf', REPO], check=True)
    subprocess.run(['cp', '-r', found, REPO], check=True)
    found = REPO
REPO = found
print('repo:', REPO)
os.chdir(REPO)
sys.path.insert(0, REPO)

locks = sorted(glob.glob(f'{SVLM_ROOT}/env/lock-colab-*.txt'))
REQ = locks[-1] if locks else 'requirements/colab.txt'
print('installing from', REQ)
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', REQ], check=True)
# Colab preinstalls an old torchao; recent peft refuses to build LoRA layers while it is present (unused here)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-q', '-y', 'torchao'], check=False)

import torch
assert torch.cuda.is_available(), 'No GPU: Runtime > Change runtime type > T4 GPU'
p = torch.cuda.get_device_properties(0)
print(f'GPU: {p.name} | compute {p.major}.{p.minor} | {p.total_memory/1e9:.1f} GB | bf16: {torch.cuda.is_bf16_supported(including_emulation=False)}')
'''


def md(s):
    return nbf.v4.new_markdown_cell(s.strip())


def code(s):
    return nbf.v4.new_code_cell(s.strip())


def notebook(cells, gpu="T4"):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "accelerator": "GPU",
        "colab": {"gpuType": gpu, "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


def smoke_test():
    return notebook([
        md("""
# 00 · Environment smoke test (milestone M0)

Run this once before any real training. It answers four questions on the actual Colab GPU:

1. Do the libraries install and do the unit tests pass?
2. Does the 4-bit teacher load on the T4, and how fast does it generate?
3. Does the **whole pipeline** run end to end with real models on a tiny dataset?
4. How long will the full `msci-code-fast` run take, in hours and compute units?

It also writes a version lock file to Drive so later sessions install exactly the same packages.

**Before you start:** *Runtime → Change runtime type → T4 GPU*. Copy the `svlm-catalogue` folder to
`MyDrive/svlm/` (or upload it as a zip when asked). Expected cost: about 30–45 minutes, roughly 1 compute unit.
"""),
        code(SETUP),
        md("## 1 · Unit tests (CPU logic: FIM spans, scoring, KD loss, data alignment)"),
        code("!python -m pytest -q tests/test_core.py"),
        md("""
## 2 · Teacher on the T4

Loads the teacher in 4-bit (bitsandbytes NF4, fp16 compute), checks memory, and measures batched
generation throughput on real FIM prompts.
"""),
        code(r'''
import subprocess, sys
# separate process: when it exits the GPU is guaranteed empty for the pipeline below
r = subprocess.run([sys.executable, 'tools/bench_teacher.py', '--config', 'configs/code_fast.yaml'],
                   capture_output=True, text=True)
print(r.stdout); print(r.stderr[-3000:] if r.returncode else '')
r.check_returncode()
TEACHER_TOK_PER_S = json.loads(r.stdout.strip().splitlines()[-1])['tokens_per_s']
'''),
        md("""
## 3 · Full pipeline, tiny data, real models

Runs all nine steps with 96 training and 32 held-out samples (`version=0.0.0`, `tier=smoke`), so it
never touches the real run. Gates will probably fail at this size; `register.force=true` lets the
register step run anyway so we can check the model card and catalogue entry.
"""),
        code(r'''
!python pipelines/run_local.py --config configs/code_fast.yaml \
    --set version=0.0.0 --set tier=smoke \
    --set data.fim.n_train=96 --set data.fim.n_heldout=32 \
    --set eval.humaneval=false --set register.force=true
'''),
        md("## 4 · Time and compute-unit estimate for the full run"),
        code(r'''
from pathlib import Path
import yaml
from svlm.utils import read_json

cfg = yaml.safe_load(open('configs/code_fast.yaml'))   # full-run settings to scale up to

CU_PER_HOUR = 1.5   # read the real figure from Colab's Resources panel and edit this

smoke = Path(SVLM_ROOT) / 'runs' / 'msci-code-fast' / '0.0.0-smoke'
sec = {s: read_json(smoke / s / '_SUCCESS')['seconds'] for s in
       ['prepare', 'teacher_generate', 'verify', 'teacher_logits', 'train_sft', 'train_kd', 'evaluate', 'quantise', 'register']}
f = cfg['data']['fim']
n_tr_s, n_he_s = 96, 32
n_tr, n_he = f['n_train'], f['n_heldout']
aug = cfg['data']['teacher_augment_fraction']
scale = {
    'prepare': 1.5,                                                      # mostly cloning, roughly fixed
    'teacher_generate': (n_he + aug * n_tr) / (n_he_s + aug * n_tr_s),
    'verify': n_tr / n_tr_s,
    'teacher_logits': n_tr / n_tr_s,
    'train_sft': n_tr / n_tr_s,
    'train_kd': n_tr / n_tr_s,
    'evaluate': n_he / n_he_s,
    'quantise': 1.0,
    'register': 1.0,
}
est = {s: sec[s] * scale[s] / 3600 for s in sec}
for s in sec:
    print(f'{s:17s} smoke {sec[s]:7.0f}s   full est. {est[s]:5.2f} h')
total = sum(est.values()) + 0.5  # + HumanEval+ run
print(f'\nEstimated full msci-code-fast run: {total:.1f} GPU hours ≈ {total*CU_PER_HOUR:.0f} compute units')
print('Model fixed costs (loading) are included in the smoke timings, so this is a slight over-estimate.')
'''),
        md("## 5 · Check the smoke model card"),
        code(r'''
from IPython.display import Markdown, display
card = Path(SVLM_ROOT) / 'model_store' / 'msci-code-fast' / '0.0.0-smoke' / 'model_card.md'
display(Markdown(card.read_text()))
'''),
        md("## 6 · Write the version lock file (later sessions install exactly these versions)"),
        code(r'''
import datetime, importlib.metadata as md_
pkgs = ['transformers', 'peft', 'accelerate', 'bitsandbytes', 'datasets', 'pyyaml', 'numpy', 'sentencepiece', 'evalplus', 'pytest']
lock = [f'{p}=={md_.version(p)}' for p in pkgs]
env_dir = Path(SVLM_ROOT) / 'env'; env_dir.mkdir(exist_ok=True)
path = env_dir / f'lock-colab-{datetime.date.today():%Y%m%d}.txt'
path.write_text('\n'.join(lock) + '\n')
info = {'gpu': p.name, 'capability': f'{p.major}.{p.minor}', 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'teacher_tokens_per_s': round(TEACHER_TOK_PER_S), 'lock': lock}
(env_dir / 'env_report.json').write_text(json.dumps(info, indent=2))
print(path.read_text()); print(json.dumps(info, indent=2))
'''),
        md("""
## Optional · vLLM teacher backend

vLLM is faster for long generations (it matters for `msci-reason` in M3) but it installs its own torch
build. Test it in a **fresh runtime** (*Runtime → Disconnect and delete runtime*), then run:

```python
!pip install -q -r requirements/vllm.txt
from vllm import LLM, SamplingParams
llm = LLM('Qwen/Qwen2.5-Coder-7B', quantization='bitsandbytes', dtype='half', max_model_len=2048, gpu_memory_utilization=0.85)
print(llm.generate(['<|fim_prefix|>def add(a, b):\n<|fim_suffix|>\n<|fim_middle|>'], SamplingParams(max_tokens=32))[0].outputs[0].text)
```

If that works, set `teacher.backend: vllm` for the teacher_generate step only, and restart the runtime
before training.

## Done when

- [ ] unit tests pass
- [ ] teacher loads and generates; throughput recorded
- [ ] all nine steps finish on the smoke run
- [ ] lock file written to `MyDrive/svlm/env/`
"""),
    ])


def code_fast():
    def step(name, explain, after=""):
        cells = [md(explain), code(f"!python pipelines/run_local.py --config configs/code_fast.yaml --steps {name}")]
        if after:
            cells.append(code(after))
        return cells

    cells = [
        md("""
# 01 · msci-code-fast (milestone M1)

Inline code completion model. **Student** Qwen2.5-Coder-0.5B, **teacher** Qwen2.5-Coder-7B in 4-bit,
trained on fill-in-the-middle spans cut from pinned, permissively licensed Python repositories.

| Step | What happens | T4 time (est.) |
|---|---|---|
| prepare | clone repos, cut FIM spans, decontaminate, split by file | ~10 min |
| teacher_generate | teacher fills held-out holes (ceiling) + 25% of train holes | ~1–2 h |
| verify | keep teacher completions that parse and stay close to the original | ~5 min |
| teacher_logits | teacher's top-20 next-token probabilities on every target | ~30–60 min |
| train_sft | LoRA fine-tune on original + accepted teacher spans | ~20–30 min |
| train_kd | continue with CE + top-k KL against the teacher | ~20–30 min |
| evaluate | base vs SFT vs final vs teacher, latency, HumanEval+, gates | ~20–30 min |
| quantise | merge LoRA, export GGUF q8_0 for CPU serving | ~5 min |
| register | copy to the model store, write model card, update catalogue.yaml | <1 min |

Each step is resumable: if Colab disconnects, re-run the setup cell and the same step cell.
Use the timings from notebook 00 to plan sessions.
"""),
        code(SETUP),
        md("## Configuration\nChange values with `--set key=value` on any step, or edit `configs/code_fast.yaml`."),
        code("from pathlib import Path\nfrom svlm.utils import read_json, read_jsonl\nRUN = Path(SVLM_ROOT) / 'runs' / 'msci-code-fast' / '0.1.0-t4'\nprint(open('configs/code_fast.yaml').read())\n!python pipelines/run_local.py --config configs/code_fast.yaml --status"),
        *step("prepare", "## 1 · prepare", r'''
print(json.dumps(read_json(RUN / 'prepare' / 'manifest.json'), indent=2))
s = read_jsonl(RUN / 'prepare' / 'train.jsonl')[0]
print('--- example', s['span_type'], s['path'], '---')
print(s['prefix'][-400:] + '\033[92m' + s['middle'] + '\033[0m' + s['suffix'][:200])
'''),
        *step("teacher_generate", "## 2 · teacher_generate\nThe slowest step. Shards of 200 are saved to Drive as they finish."),
        *step("verify", "## 3 · verify", "print(json.dumps(read_json(RUN / 'verify' / 'filter_report.json'), indent=2))"),
        *step("teacher_logits", "## 4 · teacher_logits"),
        *step("train_sft", "## 5 · train_sft", r'''
import matplotlib.pyplot as plt
def plot(step, keys=('loss',)):
    h = read_json(RUN / step / 'train_log.json')['log_history']
    for k in keys:
        pts = [(r['step'], r[k]) for r in h if k in r]
        if pts: plt.plot(*zip(*pts), label=k)
    plt.xlabel('step'); plt.legend(); plt.title(step); plt.grid(alpha=.3); plt.show()
plot('train_sft')
'''),
        *step("train_kd", "## 6 · train_kd\nLoss = 0.5 × cross-entropy + 0.5 × KL to the teacher's top-20 distribution.",
              "plot('train_kd', keys=('loss', 'ce', 'kl'))"),
        *step("evaluate", "## 7 · evaluate", r'''
m = read_json(RUN / 'evaluate' / 'metrics.json')
rows = [(k, m[k]['exact_match'], m[k]['edit_similarity'], m[k]['parse_rate']) for k in ('teacher', 'base', 'sft', 'final') if m.get(k)]
print(f"{'model':8s} {'exact':>7s} {'edit-sim':>9s} {'parse':>7s}")
for r in rows: print(f'{r[0]:8s} {r[1]:7.3f} {r[2]:9.3f} {r[3]:7.3f}')
print('\nlatency:', m['final'].get('latency')); print('humaneval:', m.get('humaneval'))
print('\nGATES:', m['gates']['verdict'].upper())
for k, v in m['gates']['checks'].items(): print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}: {v['value']} (threshold {v.get('threshold', v.get('base'))})")
'''),
        *step("quantise", "## 8 · quantise"),
        *step("register", "## 9 · register\nFails on purpose if the gates failed. Fix the cause and re-run, or add `--set register.force=true` to store it as a non-live version.", r'''
from IPython.display import Markdown, display
display(Markdown((Path(SVLM_ROOT) / 'model_store' / 'msci-code-fast' / '0.1.0-t4' / 'model_card.md').read_text()))
print(open(Path(SVLM_ROOT) / 'model_store' / 'catalogue.yaml').read())
'''),
        md("""
## 10 · Try it

Quick completion with the merged model. The gateway serves the GGUF file on CPU with llama.cpp in milestone M5.
"""),
        code(r'''
from transformers import AutoModelForCausalLM, AutoTokenizer
from svlm.fim import fim_prompt, trim_completion
mdir = RUN / 'merged'
tok = AutoTokenizer.from_pretrained(mdir); model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.float16).cuda()
prefix = "import numpy as np\n\ndef sharpe_ratio(returns, rf=0.0, periods=252):\n"
suffix = "\n    return mean / std * np.sqrt(periods)\n"
ids = tok(fim_prompt(prefix, suffix), return_tensors='pt').to('cuda')
out = model.generate(**ids, max_new_tokens=64, do_sample=False)
print(prefix + '\033[92m' + trim_completion(tok.decode(out[0][ids['input_ids'].shape[1]:]), 'block', suffix) + '\033[0m' + suffix)
'''),
        md("""
## Get the model onto your machine

The release folder is `MyDrive/svlm/model_store/msci-code-fast/0.1.0-t4/` (GGUF, merged weights, model card,
metrics, lineage) plus `MyDrive/svlm/model_store/catalogue.yaml`. With Google Drive for Desktop it syncs to
your computer automatically; copy it into the gateway server's model store (for example
`C:\\projects\\TechFest\\model_store\\`).
"""),
    ]
    return notebook(cells)


def code_chat():
    """02 · msci-code: one coder for inline completion and chat (L4, vLLM teacher)."""
    def step(name, explain, after="", extra=""):
        cells = [md(explain), code(f"!python pipelines/run_local.py --config configs/code.yaml {{SETS}} --steps {name}{extra}")]
        if after:
            cells.append(code(after))
        return cells

    cells = [
        md("""
# 02 · msci-code: one coder for completion and chat

One **Qwen2.5-Coder-1.5B** student does both jobs in the IDE: inline completion (fill-in-the-middle) and
chat (write, explain, document and fix Python). The teacher is **Qwen2.5-Coder-14B-Instruct-AWQ** on vLLM.
`msci-code-fast` stays live for autocomplete until this model beats it on the same FIM spans.

| SFT mix | Source | Target | Check |
|---|---|---|---|
| FIM (~2,500) | spans from the pinned repos, same as msci-code-fast | the original code | ground truth |
| MBPP (~380) | MBPP train problems | teacher, 4 answers | MBPP's own tests |
| synth (~2,000 seeds) | problems the teacher writes from real repo functions | teacher, 4 answers | reference passes its tests **and** ≥ 2 answers agree |
| docstring (~600) | repo functions with docstrings removed | the original function | ground truth |
| explain (~400) | repo functions | teacher | length rules |

**Runtime: L4** (Runtime → Change runtime type → L4 GPU, High RAM on). Time estimates below are guesses
until sections 2–3 measure them:

| Step | What happens | L4 time (est.) |
|---|---|---|
| prepare | clone repos, FIM spans (identical to msci-code-fast), MBPP, repo functions | ~10 min |
| teacher_generate | vLLM: write problems, 4 answers per problem, explanations | **section 3 projects it** (2–5 h guess) |
| verify | run every answer against its tests in a sandbox | ~10–20 min |
| teacher_logits / train_kd | skipped (`train.kd.enabled: false`; distillation hurt msci-code-fast) | – |
| train_sft | bf16 LoRA r=32, 2 epochs | ~1 h |
| evaluate | FIM (vs msci-code-fast) + chat pass@1 for base / SFT; gates | ~20 min |
| quantise | merge, GGUF q8_0, CPU latency of the GGUF | ~10 min |
| register | model store, model card, catalogue.yaml | <1 min |

Every step resumes after a disconnect: re-run the setup cell, the variables cell and the same step cell.
"""),
        code(SETUP),
        md("""
## 0 · vLLM

vLLM brings its own torch build. Install it once per runtime, then check that training libraries still import.
If the install fails or breaks the check below, use the slow fallback instead: in section 4 add
`--set teacher.backend=hf` (4-bit transformers; expect several times longer teacher steps).
"""),
        code(r'''
!pip install -q -r requirements/vllm.txt
!python -c "import vllm, torch, transformers, peft; print('vllm', vllm.__version__, '| torch', torch.__version__, '| cuda', torch.cuda.is_available(), '| transformers', transformers.__version__)"
'''),
        md("## 1 · Unit tests (CPU: chat format, sandbox, problem parsing, verify, gates)"),
        code("!python -m pytest -q tests/test_core.py tests/test_code_task.py"),
        md("""
## 2 · Pick the starting checkpoint

Scores the **untrained** Base and Instruct 1.5B coders on msci-code-fast's held-out FIM spans and on MBPP
validation (tests run in the sandbox). Training restores FIM easily (2,500 original spans in the mix); chat
ability is harder to create from nothing, so prefer the better chat score unless its FIM score is far behind.
~10 min.
"""),
        code("!python tools/compare_bases.py --config configs/code.yaml --n-fim 150"),
        code(r'''
# Set the starting checkpoint from the table above, then run this cell (and re-run it after any reconnect).
STUDENT = 'Qwen/Qwen2.5-Coder-1.5B-Instruct'
VERSION = '0.1.0'
SETS = f'--set student.model={STUDENT} --set version={VERSION}'
from pathlib import Path
from svlm.utils import read_json, read_jsonl
RUN = Path(SVLM_ROOT) / 'runs' / 'msci-code' / f'{VERSION}-l4'
print(SETS)
'''),
        md("""
## 3 · Teacher throughput

Loads the 14B AWQ teacher in vLLM in its own process, answers 48 MBPP problems 4 times, and projects the
full teacher step for this config. Use the projection to plan the session (Colab Pro sessions last up to 24 h,
and every shard is saved, so a disconnect costs at most one shard).
"""),
        code("!python tools/bench_chat_teacher.py --config configs/code.yaml --problems 48"),
        md("""
## 4 · Smoke run (tiny data, real models, ~30–40 min)

All steps with a few dozen examples per type, as `0.0.0-smoke`, so nothing touches the real run. Gates
will fail at this size (and `fim_vs_reference` cannot compare: the smoke held-out spans differ);
`register.force=true` registers it as non-live so the model card can be checked.
"""),
        code(r'''
SMOKE = ('--set version=0.0.0 --set tier=smoke --set data.fim.n_train=200 --set data.fim.n_heldout=40 '
         '--set data.code.fim_train=150 --set data.code.mbpp_max=40 --set data.code.synth_train=60 '
         '--set data.code.synth_heldout=20 --set data.code.docstring=30 --set data.code.explain=20 '
         '--set train.sft.epochs=1 --set register.force=true')
!python pipelines/run_local.py --config configs/code.yaml --set student.model={STUDENT} {SMOKE}
'''),
        code(r'''
smoke = Path(SVLM_ROOT) / 'model_store' / 'msci-code' / '0.0.0-smoke' / 'model_card.md'
from IPython.display import Markdown, display
display(Markdown(smoke.read_text()))
'''),
        md("""
## 5 · Full run

One cell per step. `teacher_generate` runs vLLM in a child process, so GPU memory is free again for training
in the same session.
"""),
        code("!python pipelines/run_local.py --config configs/code.yaml {SETS} --status"),
        *step("prepare", "### prepare", r'''
m = read_json(RUN / 'prepare' / 'manifest.json')
print(json.dumps({k: m[k] for k in ('fim', 'chat_train', 'chat_heldout', 'decontam_dropped')}, indent=2))
ex = [r for r in read_jsonl(RUN / 'prepare' / 'chat_train.jsonl') if r['type'] == 'synth_seed'][0]
print('--- a repository function that seeds a teacher-written problem:', ex['path']); print(ex['code'])
'''),
        *step("teacher_generate", "### teacher_generate\nThe long step. Shards of 256 are saved to Drive as they finish."),
        *step("verify", "### verify\nEvery teacher answer is run against its tests in a sandbox.", r'''
r = read_json(RUN / 'verify' / 'filter_report.json')
print(json.dumps({k: r[k] for k in ('teacher_train_filter', 'sft_mix', 'chat_eval', 'teacher_heldout')}, indent=2))
row = next(x for x in read_jsonl(RUN / 'verify' / 'sft.jsonl') if x.get('type') == 'synth')
print('--- user ---'); print(row['messages'][0]['content']); print('--- assistant (kept: passed the tests) ---'); print(row['messages'][1]['content'])
'''),
        *step("teacher_logits", "### teacher_logits\nSkipped unless `train.kd.enabled=true`."),
        *step("train_sft", "### train_sft", r'''
import matplotlib.pyplot as plt
h = read_json(RUN / 'train_sft' / 'train_log.json')['log_history']
pts = [(r['step'], r['loss']) for r in h if 'loss' in r]
plt.plot(*zip(*pts)); plt.xlabel('step'); plt.ylabel('loss'); plt.title('train_sft'); plt.grid(alpha=.3); plt.show()
'''),
        *step("train_kd", "### train_kd\nSkipped unless `train.kd.enabled=true`."),
        *step("evaluate", "### evaluate", r'''
m = read_json(RUN / 'evaluate' / 'metrics.json')
ref = m.get('fim_reference', {})
print('FIM (same held-out spans as msci-code-fast)')
if ref.get('status') == 'ok': print(f"  reference msci-code-fast:{ref['version']}  edit-sim {ref['edit_similarity']:.3f}")
else: print('  reference:', ref.get('status'))
for k in ('base', 'sft', 'final'):
    f = m[k]['fim']; print(f"  {k:6s} exact {f['exact_match']:.3f}  edit-sim {f['edit_similarity']:.3f}  parse {f['parse_rate']:.3f}")
print('Chat pass@1 (held-out problems, tests in sandbox)')
print(f"  teacher {m['teacher']['chat']['pass@1']}  {m['teacher']['chat']['by_type']}")
for k in ('base', 'sft', 'final'): print(f"  {k:7s} {m[k]['chat']['pass@1']}  {m[k]['chat']['by_type']}")
print('\nGATES:', m['gates']['verdict'].upper())
for k, v in m['gates']['checks'].items(): print(f"  {'PASS' if v['pass'] else 'FAIL'}  {k}: {v['value']} (threshold {v.get('threshold', v.get('base'))})")
'''),
        *step("quantise", "### quantise\nAlso measures time to first token of the GGUF on this machine's CPU (indicative: Colab's CPU is not your server).",
              "print(json.dumps(read_json(RUN / 'quantise' / 'summary.json').get('cpu_latency'), indent=2))"),
        *step("register", "### register\nFails on purpose if a gate failed; add `--set register.force=true` to store it as non-live.", r'''
from IPython.display import Markdown, display
display(Markdown((Path(SVLM_ROOT) / 'model_store' / 'msci-code' / f'{VERSION}-l4' / 'model_card.md').read_text()))
print(open(Path(SVLM_ROOT) / 'model_store' / 'catalogue.yaml').read())
'''),
        md("## 6 · Try it: chat and inline completion with the merged model"),
        code(r'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from svlm.chat import format_chat, trim_reply
from svlm.fim import fim_prompt, trim_completion
mdir = RUN / 'merged'
tok = AutoTokenizer.from_pretrained(mdir); model = AutoModelForCausalLM.from_pretrained(mdir, dtype=torch.bfloat16).cuda()

def ask(question, max_new_tokens=400):
    ids = tok(format_chat([{'role': 'user', 'content': question}]), return_tensors='pt', add_special_tokens=False).to('cuda')
    out = model.generate(**ids, max_new_tokens=max_new_tokens, do_sample=False)
    return trim_reply(tok.decode(out[0][ids['input_ids'].shape[1]:]))

print(ask('Write a function that returns the maximum drawdown of a pandas Series of prices.'))
print('\n' + '=' * 60 + '\n')
prefix = "import numpy as np\n\ndef sharpe_ratio(returns, rf=0.0, periods=252):\n"
suffix = "\n    return mean / std * np.sqrt(periods)\n"
ids = tok(fim_prompt(prefix, suffix), return_tensors='pt', add_special_tokens=False).to('cuda')
out = model.generate(**ids, max_new_tokens=64, do_sample=False)
print(prefix + '\033[92m' + trim_completion(tok.decode(out[0][ids['input_ids'].shape[1]:]), 'block', suffix) + '\033[0m' + suffix)
'''),
        md("""
## 7 · Serve it from your machine

1. Copy `MyDrive/svlm/model_store/` (the new `msci-code/` folder and `catalogue.yaml`) into
   `C:\\projects\\TechFest\\model_store\\`.
2. Restart the gateway. It loads every live model in `catalogue.yaml`: `msci-code-fast` and `msci-code`.
3. In `~/.continue/config.yaml`, add the `msci-code` entry from `gateway/continue-config.yaml` (chat, edit,
   apply). Move `autocomplete` to it only if its CPU latency is acceptable for you; `msci-code-fast` is about
   3× faster.

## Done when

- [ ] vLLM installs and the training libraries still import
- [ ] the starting checkpoint is chosen from section 2
- [ ] the teacher projection fits your compute budget
- [ ] the smoke run finishes all steps and its model card renders
- [ ] the full run passes its gates (or the failures are understood) and is registered
"""),
    ]
    return notebook(cells, gpu="L4")


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    nbf.write(smoke_test(), OUT / "00_smoke_test.ipynb")
    nbf.write(code_fast(), OUT / "01_code_fast.ipynb")
    nbf.write(code_chat(), OUT / "02_code.ipynb")
    print("wrote", sorted(p.name for p in OUT.glob("*.ipynb")))

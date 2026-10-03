"""End-to-end CPU dry run with tiny random models (no model downloads, ~2-4 minutes).

Builds a small BPE tokenizer with the Qwen FIM special tokens, two tiny random Qwen2 models
(teacher bigger than student, same tokenizer), then runs every pipeline step on a few dozen
samples from one real repository. It checks wiring, file formats, resume markers and the
model card, not model quality.

    python tests/e2e_tiny.py [workdir]
"""
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

SPECIAL = ["<|endoftext|>", "<|fim_prefix|>", "<|fim_middle|>", "<|fim_suffix|>", "<|fim_pad|>", "<|file_sep|>", "<|repo_name|>"]


def make_tiny_models(work: Path, corpus_dir: Path) -> tuple[Path, Path]:
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen2Config, Qwen2ForCausalLM

    files = [str(p) for p in corpus_dir.rglob("*.py")][:200]
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    tk.train(files, trainers.BpeTrainer(vocab_size=2000, special_tokens=SPECIAL, show_progress=False))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|endoftext|>", pad_token="<|endoftext|>")
    tok.add_special_tokens({"additional_special_tokens": SPECIAL[1:]})

    out = []
    for name, hidden, layers in (("student", 64, 2), ("teacher", 128, 3)):
        cfg = Qwen2Config(vocab_size=len(tok), hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=layers,
                          num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=2048,
                          eos_token_id=tok.eos_token_id, pad_token_id=tok.pad_token_id, tie_word_embeddings=True)
        d = work / f"tiny-{name}"
        Qwen2ForCausalLM(cfg).save_pretrained(d)
        tok.save_pretrained(d)
        out.append(d)
    return out[0], out[1]


def main():
    work = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="svlm-e2e-"))
    work.mkdir(parents=True, exist_ok=True)
    corpus = work / "cache" / "repos" / "requests-v2.32.3"
    if not corpus.exists():
        subprocess.run(["git", "clone", "--depth", "1", "--branch", "v2.32.3", "https://github.com/psf/requests", str(corpus)], check=True)
    student, teacher = make_tiny_models(work, corpus)

    sets = [
        f"paths.root={work}", "version=0.0.1", "tier=cpu-e2e",
        f"student.model={student}", f"teacher.model={teacher}", "teacher.load_in_4bit=false",
        "data.sources=[{repo: https://github.com/psf/requests, ref: v2.32.3, license: Apache-2.0, include: ['src/requests/*.py']}]",
        "data.fim.n_train=60", "data.fim.n_heldout=20", "data.decontam.eval_sets=[]",
        "teacher.gen.max_new_tokens=16", "teacher.gen.batch_size=8", "teacher.logits.batch_size=4",
        "train.use_unsloth=false",
        "train.sft.per_device_batch=4", "train.sft.grad_accum=1", "train.sft.save_steps=5",
        "train.kd.per_device_batch=4", "train.kd.grad_accum=1", "train.kd.save_steps=5",
        "eval.max_new_tokens=16", "eval.gen_batch_size=8", "eval.humaneval=false", "eval.latency_prompts=6",
        "quantise.gguf_outtype=null", "register.force=true",
    ]
    cmd = [sys.executable, str(REPO / "pipelines" / "run_local.py"), "--config", str(REPO / "configs" / "code_fast.yaml")]
    for s in sets:
        cmd += ["--set", s]
    r = subprocess.run(cmd)
    if r.returncode:
        raise SystemExit("pipeline failed")

    # second invocation must skip everything (resume markers)
    r2 = subprocess.run(cmd, capture_output=True, text=True)
    assert r2.returncode == 0 and r2.stderr.count("skip ") == 9, r2.stderr[-2000:]

    store = work / "model_store"
    card = store / "msci-code-fast" / "0.0.1-cpu-e2e" / "model_card.md"
    assert card.exists(), "model card missing"
    assert "n/a" not in card.read_text().split("## Results")[1].split("Latency")[0], "results table incomplete"
    assert (store / "catalogue.yaml").exists()
    print("\nE2E OK. Work dir:", work)
    print(card.read_text()[:1500])


if __name__ == "__main__":
    main()

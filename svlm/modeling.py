"""Model loading, LoRA setup, batched generation and merging.

All T4-specific choices live here: fp16 everywhere (no bf16 on sm_75), bitsandbytes NF4 for
4-bit loading, SDPA attention (no FlashAttention-2 on T4).
"""
from __future__ import annotations

import gc
from pathlib import Path

import torch

from .utils import log


def gpu_info() -> dict:
    if not torch.cuda.is_available():
        return {"cuda": False}
    p = torch.cuda.get_device_properties(0)
    return {
        "cuda": True,
        "name": p.name,
        "capability": f"{p.major}.{p.minor}",
        "memory_gb": round(p.total_memory / 1e9, 1),
        "bf16_supported": torch.cuda.is_bf16_supported(including_emulation=False),
    }


def compute_dtype() -> torch.dtype:
    if not torch.cuda.is_available():
        return torch.float32  # CPU tests / dry runs
    # including_emulation=False: recent torch reports bf16 on T4 (sm_75) via slow emulation
    if torch.cuda.is_bf16_supported(including_emulation=False):
        return torch.bfloat16  # L4/A100: use bf16 automatically
    return torch.float16       # T4


def free_gpu(*objs) -> None:
    """Collect garbage and return cached CUDA memory to the driver.

    Passing objects here cannot free them: the caller still holds its own reference. Drop those
    first (`model = None` / `del model`); otherwise the weights stay resident and the next
    `device_map="auto"` load sees a full GPU and tries to offload to CPU.
    """
    del objs
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def load_tokenizer(name: str, padding_side: str = "left"):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(name)
    tok.padding_side = padding_side
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def load_causal_lm(name: str, load_in_4bit: bool = False, device_map: str | dict = "auto"):
    """Plain transformers loading (teacher, eval)."""
    from transformers import AutoModelForCausalLM, BitsAndBytesConfig

    dtype = compute_dtype()
    kwargs: dict = {"dtype": dtype, "device_map": device_map, "attn_implementation": "sdpa"}
    if load_in_4bit:
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=dtype,
            bnb_4bit_use_double_quant=True,
        )
    log.info("loading %s (4bit=%s, dtype=%s)", name, load_in_4bit, dtype)
    model = AutoModelForCausalLM.from_pretrained(name, **kwargs)
    model.eval()
    return model


def _want_unsloth(setting) -> bool:
    if setting in (False, "false"):
        return False
    try:
        import unsloth  # noqa: F401
        return True
    except Exception as e:
        if setting in (True, "true"):
            raise RuntimeError(f"use_unsloth=true but unsloth failed to import: {e}")
        log.info("unsloth not available (%s); using peft", e)
        return False


def load_student_for_training(cfg: dict, adapter_dir: str | Path | None = None):
    """Return (model, tokenizer) with trainable LoRA. Resumes from `adapter_dir` if given."""
    s = cfg["student"]
    lora = cfg["train"]["sft"]
    max_len = s.get("max_seq_len", 1024)

    if _want_unsloth(cfg["train"].get("use_unsloth", "auto")):
        from unsloth import FastLanguageModel

        model, tok = FastLanguageModel.from_pretrained(
            model_name=str(adapter_dir) if adapter_dir else s["model"],
            max_seq_length=max_len,
            dtype=compute_dtype(),
            load_in_4bit=s.get("load_in_4bit", False),
        )
        if adapter_dir is None:
            model = FastLanguageModel.get_peft_model(
                model,
                r=lora["lora_r"],
                lora_alpha=lora["lora_alpha"],
                lora_dropout=lora["lora_dropout"],
                target_modules=lora["target_modules"],
                use_gradient_checkpointing="unsloth",
                random_state=cfg.get("seed", 42),
            )
        tok.padding_side = "right"
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        return model, tok

    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training

    tok = load_tokenizer(s["model"], padding_side="right")
    model = load_causal_lm(s["model"], load_in_4bit=s.get("load_in_4bit", False), device_map={"": 0} if torch.cuda.is_available() else None)
    if s.get("load_in_4bit", False):
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True)
    else:
        model.gradient_checkpointing_enable()
        model.enable_input_require_grads()
    if adapter_dir:
        model = PeftModel.from_pretrained(model, str(adapter_dir), is_trainable=True)
    else:
        model = get_peft_model(
            model,
            LoraConfig(
                r=lora["lora_r"],
                lora_alpha=lora["lora_alpha"],
                lora_dropout=lora["lora_dropout"],
                target_modules=lora["target_modules"],
                task_type="CAUSAL_LM",
            ),
        )
    # fp16 AMP needs fp32 master weights for the trainable parameters
    for p in model.parameters():
        if p.requires_grad and p.dtype != torch.float32:
            p.data = p.data.float()
    model.print_trainable_parameters()
    return model, tok


def load_student_for_inference(cfg: dict, adapter_dir: str | Path | None):
    """Base student plus (optionally) a LoRA adapter, ready for generate()."""
    s = cfg["student"]
    tok = load_tokenizer(s["model"], padding_side="left")
    model = load_causal_lm(s["model"], load_in_4bit=s.get("load_in_4bit", False))
    if adapter_dir:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tok


@torch.no_grad()
def generate_batch(model, tok, prompts: list[str], max_new_tokens: int, temperature: float = 0.0) -> list[str]:
    enc = tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(model.device)
    kwargs = dict(max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
    if temperature and temperature > 0:
        kwargs.update(do_sample=True, temperature=temperature, top_p=0.95)
    else:
        kwargs.update(do_sample=False)
    out = model.generate(**enc, **kwargs)
    new = out[:, enc["input_ids"].shape[1] :]
    return tok.batch_decode(new, skip_special_tokens=False)


def merge_and_save(cfg: dict, adapter_dir: str | Path, out_dir: str | Path) -> Path:
    """Merge LoRA into fp16 base weights and save a standalone HF model directory."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM

    out_dir = Path(out_dir)
    if (out_dir / "config.json").exists():
        return out_dir
    base = AutoModelForCausalLM.from_pretrained(cfg["student"]["model"], dtype=torch.float16)
    merged = PeftModel.from_pretrained(base, str(adapter_dir)).merge_and_unload()
    merged.save_pretrained(out_dir, safe_serialization=True)
    load_tokenizer(cfg["student"]["model"]).save_pretrained(out_dir)
    base = merged = None
    free_gpu()
    log.info("merged model saved to %s", out_dir)
    return out_dir

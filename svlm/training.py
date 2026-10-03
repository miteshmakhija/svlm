"""Shared Trainer setup for SFT and KD (transformers Trainer, LoRA, resumable checkpoints)."""
from __future__ import annotations

from pathlib import Path

import torch

from .data import PadCollator
from .kd_loss import kd_loss
from .modeling import compute_dtype
from .utils import log


def training_args(out_dir: Path, hp: dict, seed: int, n_examples: int):
    from transformers import TrainingArguments

    bf16 = compute_dtype() == torch.bfloat16
    steps_per_epoch = max(1, n_examples // (hp["per_device_batch"] * hp["grad_accum"]))
    return TrainingArguments(
        output_dir=str(out_dir),
        per_device_train_batch_size=hp["per_device_batch"],
        gradient_accumulation_steps=hp["grad_accum"],
        num_train_epochs=hp.get("epochs", 1),
        learning_rate=hp["learning_rate"],
        warmup_steps=int(hp.get("warmup_ratio", 0.03) * steps_per_epoch * hp.get("epochs", 1)),
        lr_scheduler_type="cosine",
        weight_decay=0.0,
        max_grad_norm=1.0,
        fp16=torch.cuda.is_available() and not bf16,
        bf16=bf16,
        logging_steps=10,
        save_strategy="steps",
        save_steps=hp.get("save_steps", 100),
        save_total_limit=2,
        report_to=[],
        remove_unused_columns=False,
        dataloader_num_workers=0,
        seed=seed,
        optim="adamw_torch",
    )


def _latest_checkpoint(out_dir: Path) -> str | None:
    cks = sorted(out_dir.glob("checkpoint-*"), key=lambda p: int(p.name.split("-")[-1]))
    return str(cks[-1]) if cks else None


class KDTrainer:
    """Factory for a Trainer subclass with the top-k KD loss (imported lazily)."""

    @staticmethod
    def build(alpha: float, temperature: float):
        from transformers import Trainer

        class _T(Trainer):
            def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
                topk_ids = inputs.pop("topk_ids")
                topk_lp = inputs.pop("topk_logprobs")
                labels = inputs.pop("labels")
                out = model(input_ids=inputs["input_ids"], attention_mask=inputs["attention_mask"])
                logits = out.logits[:, :-1]
                lab = labels[:, 1:]
                m = lab != -100
                loss, parts = kd_loss(
                    logits[m], lab[m], topk_ids[:, 1:][m], topk_lp[:, 1:][m], alpha=alpha, temperature=temperature
                )
                if self.state.global_step % 10 == 0:
                    self.log({"ce": parts["ce"], "kl": parts["kl"]})
                return (loss, out) if return_outputs else loss

        return _T


def train(model, tok, feats: list[dict], out_dir: Path, hp: dict, seed: int, kd: dict | None = None, k: int | None = None) -> Path:
    from transformers import Trainer

    args = training_args(out_dir / "checkpoints", hp, seed, len(feats))
    collator = PadCollator(tok.pad_token_id, k=k if kd else None)
    cls = KDTrainer.build(kd["alpha"], kd["temperature"]) if kd else Trainer
    trainer = cls(model=model, args=args, train_dataset=feats, data_collator=collator)
    resume = _latest_checkpoint(out_dir / "checkpoints")
    if resume:
        log.info("resuming from %s", resume)
    result = trainer.train(resume_from_checkpoint=resume)
    adapter = out_dir / "adapter"
    model.save_pretrained(str(adapter))
    tok.save_pretrained(str(adapter))
    (out_dir / "train_log.json").write_text(
        __import__("json").dumps({"metrics": result.metrics, "log_history": trainer.state.log_history}, indent=2, default=str)
    )
    log.info("training finished: %s", result.metrics)
    return adapter

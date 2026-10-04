"""Chat teacher for the instruct tasks: one interface, two backends.

    teacher = ChatTeacher(cfg)          # backend from cfg["teacher"]["backend"]: vllm (default) | hf
    replies = teacher.chat(conversations, n=4, temperature=0.7, max_tokens=768)
    teacher.close()

`replies[i]` is a list of `n` strings for conversation i. Conversations use the teacher's own chat
template (it is an off-the-shelf instruct model); only the student uses svlm.chat formatting.

vLLM is the practical choice for long outputs (an AWQ 14B teacher on an L4 generates hundreds of
tokens/s across a batch). The hf backend is a slow fallback for when vLLM will not install.
"""
from __future__ import annotations

from .modeling import free_gpu, load_causal_lm, load_tokenizer
from .utils import chunks, log


class ChatTeacher:
    def __init__(self, cfg: dict):
        t = cfg["teacher"]
        self.name = t["model"]
        self.backend = t.get("backend", "vllm")
        self.seed = cfg.get("seed", 42)
        if self.backend == "vllm":
            from vllm import LLM

            kwargs = dict(model=self.name, max_model_len=t.get("max_model_len", 4096), seed=self.seed,
                          gpu_memory_utilization=t.get("gpu_memory_utilization", 0.85))
            if "awq" in self.name.lower():
                kwargs["dtype"] = "half"  # AWQ kernels run in fp16
            elif t.get("load_in_4bit"):
                kwargs["quantization"] = "bitsandbytes"
            log.info("teacher: vLLM %s", kwargs)
            self.llm = LLM(**kwargs)
        else:
            self.tok = load_tokenizer(self.name, padding_side="left")
            self.model = load_causal_lm(self.name, load_in_4bit=t.get("load_in_4bit", True))
            self.batch_size = t.get("gen", {}).get("batch_size", 8)

    def chat(self, convs: list[list[dict]], n: int = 1, temperature: float = 0.0, max_tokens: int = 768) -> list[list[str]]:
        if not convs:
            return []
        if self.backend == "vllm":
            from vllm import SamplingParams

            sp = SamplingParams(n=n, temperature=temperature, top_p=0.95 if temperature > 0 else 1.0,
                                max_tokens=max_tokens, seed=self.seed)
            res = self.llm.chat(convs, sp, use_tqdm=False)
            return [[o.text for o in r.outputs] for r in res]
        return self._hf_chat(convs, n, temperature, max_tokens)

    def _hf_chat(self, convs, n, temperature, max_tokens):
        import torch

        out: list[list[str]] = []
        for batch in chunks(convs, self.batch_size):
            prompts = [self.tok.apply_chat_template(c, tokenize=False, add_generation_prompt=True) for c in batch]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            kw = dict(max_new_tokens=max_tokens, pad_token_id=self.tok.pad_token_id, num_return_sequences=n)
            kw.update(do_sample=True, temperature=temperature, top_p=0.95) if temperature > 0 else kw.update(do_sample=False)
            if n > 1 and temperature == 0:
                raise ValueError("n > 1 needs temperature > 0")
            with torch.no_grad():
                gen = self.model.generate(**enc, **kw)
            texts = self.tok.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
            out.extend(texts[i * n:(i + 1) * n] for i in range(len(batch)))
        return out

    def close(self) -> None:
        self.llm = self.model = None
        free_gpu()

"""Chat formatting shared by training, evaluation and the gateway.

Qwen2.5 models use ChatML. We format it ourselves instead of calling a tokenizer's chat template,
so the exact same text is produced in training, in evaluation and at serving time, whichever
checkpoint (base or instruct) the student started from:

    <|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n{reply}<|im_end|>\n
"""
from __future__ import annotations

import re

IM_START = "<|im_start|>"
IM_END = "<|im_end|>"
CHAT_STOP = [IM_END, "<|endoftext|>", IM_START]
DEFAULT_SYSTEM = "You are msci-code, a careful Python coding assistant. Answer concisely and put code in ```python blocks."


def format_chat(messages: list[dict], system: str | None = DEFAULT_SYSTEM, add_generation_prompt: bool = True) -> str:
    """ChatML text for `messages` ({"role", "content"}). A leading system message overrides `system`."""
    msgs = list(messages)
    if msgs and msgs[0]["role"] == "system":
        system = msgs.pop(0)["content"]
    parts = [f"{IM_START}system\n{system}{IM_END}\n"] if system else []
    for m in msgs:
        parts.append(f"{IM_START}{m['role']}\n{m['content']}{IM_END}\n")
    if add_generation_prompt:
        parts.append(f"{IM_START}assistant\n")
    return "".join(parts)


def split_for_training(messages: list[dict], system: str | None = DEFAULT_SYSTEM) -> tuple[str, str]:
    """(prompt, target) for SFT: everything up to the last assistant turn is prompt, its content + <|im_end|> is target."""
    if not messages or messages[-1]["role"] != "assistant":
        raise ValueError("last message must be the assistant reply")
    return format_chat(messages[:-1], system, add_generation_prompt=True), messages[-1]["content"] + IM_END + "\n"


def trim_reply(text: str) -> str:
    for s in CHAT_STOP:
        i = text.find(s)
        if i != -1:
            text = text[:i]
    return text.strip()


_FENCE = re.compile(r"```(?:python|py|Python)?[ \t]*\n(.*?)```", re.S)


def extract_code(text: str) -> str:
    """The code a reply proposes: all ```python blocks joined, or the whole text if there are none."""
    blocks = _FENCE.findall(text)
    if blocks:
        return "\n\n".join(b.strip("\n") for b in blocks)
    # an unterminated fence (generation hit max tokens): take what is there
    if "```" in text:
        return text.split("```", 1)[1].split("\n", 1)[-1]
    return text

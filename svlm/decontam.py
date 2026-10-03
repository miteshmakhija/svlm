"""N-gram decontamination: drop training samples that overlap evaluation benchmarks."""
from __future__ import annotations

import re
from typing import Iterable

from .utils import log

_TOKEN = re.compile(r"\w+|[^\w\s]")

# benchmark name -> (HF dataset id, config, split, fields to index)
EVAL_SETS = {
    "humaneval": ("openai/openai_humaneval", None, "test", ["prompt", "canonical_solution", "test"]),
    "mbpp": ("google-research-datasets/mbpp", "full", "test", ["text", "code", "test_list"]),
    "gsm8k": ("openai/gsm8k", "main", "test", ["question", "answer"]),
}


def ngrams(text: str, n: int) -> set[tuple[str, ...]]:
    toks = _TOKEN.findall(text)
    return {tuple(toks[i : i + n]) for i in range(len(toks) - n + 1)}


def load_eval_texts(names: Iterable[str]) -> list[str]:
    """Download benchmark texts with `datasets`. Missing sets are skipped with a warning."""
    texts: list[str] = []
    try:
        from datasets import load_dataset
    except ImportError:
        log.warning("datasets not installed; decontamination skipped")
        return texts
    for name in names:
        if name not in EVAL_SETS:
            log.warning("unknown eval set %s", name)
            continue
        ds_id, conf, split, fields = EVAL_SETS[name]
        try:
            ds = load_dataset(ds_id, conf, split=split)
        except Exception as e:  # network or hub issue: keep going, but say so
            log.warning("could not load %s (%s); not decontaminated against it", name, e)
            continue
        for row in ds:
            parts = []
            for fld in fields:
                v = row.get(fld)
                parts.append("\n".join(v) if isinstance(v, list) else str(v or ""))
            texts.append("\n".join(parts))
        log.info("decontam: indexed %s (%d rows)", name, len(ds))
    return texts


def build_index(texts: Iterable[str], n: int) -> set[tuple[str, ...]]:
    idx: set[tuple[str, ...]] = set()
    for t in texts:
        idx |= ngrams(t, n)
    return idx


def is_contaminated(text: str, index: set[tuple[str, ...]], n: int) -> bool:
    if not index:
        return False
    return any(g in index for g in ngrams(text, n))

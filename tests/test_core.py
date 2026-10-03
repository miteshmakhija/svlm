"""Unit tests that run on CPU without downloading models."""
import math
import random
import sys
from pathlib import Path

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svlm import decontam, fim  # noqa: E402
from svlm.config import STEPS, Run, deep_merge, load_config, parse_overrides  # noqa: E402
from svlm.data import PadCollator, load_topk_shards, save_topk_shard  # noqa: E402
from svlm.kd_loss import kd_loss, topk_kl  # noqa: E402

SRC = '''import math


def area(r):
    """Circle area."""
    if r < 0:
        raise ValueError("negative radius")
    value = math.pi * r ** 2
    return round(value, 4)


class Account:
    def __init__(self, owner, balance=0.0):
        self.owner = owner
        self.balance = balance

    def deposit(self, amount):
        if amount <= 0:
            raise ValueError("amount must be positive")
        self.balance += amount
        return self.balance
'''

FILE = {"file_id": "f1", "repo": "r", "path": "a.py", "license": "MIT", "content": SRC}


# ---------- FIM ----------

def test_candidate_spans_cover_all_types_and_splice_back():
    spans = fim.candidate_spans(SRC, max_middle_lines=12, rng=random.Random(0))
    assert all(spans[t] for t in fim.SPAN_TYPES)
    for t, lst in spans.items():
        for sp in lst:
            middle = SRC[sp.start : sp.end]
            assert middle.strip()
            assert fim.parses_in_file(SRC, sp.start, sp.end, middle)  # original always parses


def test_function_body_skips_docstring():
    spans = fim.candidate_spans(SRC, 12, random.Random(0))["function_body"]
    bodies = [SRC[s.start : s.end] for s in spans]
    assert any(b.lstrip().startswith("if r < 0") for b in bodies)
    assert not any('"""Circle area."""' in b for b in bodies)


def test_build_samples_respects_mix_and_budget():
    rows = fim.build_samples([FILE], 9, {"line": 1, "block": 1, "function_body": 1}, 12, 60, 40, seed=1)
    assert rows and len({r["id"] for r in rows}) == len(rows)
    for r in rows:
        assert len(r["prefix"]) <= 60 + 80 and len(r["suffix"]) <= 40
        assert SRC[r["start"] : r["end"]] == r["middle"]
        assert SRC[: r["start"]].endswith(r["prefix"])
        assert SRC[r["end"] :].startswith(r["suffix"])


def test_fim_prompt_format():
    assert fim.fim_prompt("a", "b") == "<|fim_prefix|>a<|fim_suffix|>b<|fim_middle|>"


def test_trim_completion():
    assert fim.trim_completion("x = 1\ny = 2<|endoftext|>junk", "line", "") == "x = 1"
    suffix = "\n    return total\n"
    assert fim.trim_completion("    total += 1\n    return total\n", "block", suffix) == "    total += 1\n"


def test_scoring():
    s = fim.score("value = 1  ", {"middle": "value = 1", "start": 0, "end": 0})
    assert s["exact_match"] == 1.0 and s["edit_similarity"] == 1.0
    assert fim.edit_similarity("abc", "xyz") < 0.5
    assert not fim.parses_in_file("x = (\n", 0, 0, "")


# ---------- decontamination ----------

def test_decontam_detects_overlap():
    bench = "def add(a, b): return a + b if a is not None and b is not None else 0 end"
    idx = decontam.build_index([bench], 13)
    assert decontam.is_contaminated("prefix " + bench + " suffix", idx, 13)
    assert not decontam.is_contaminated("completely different text here", idx, 13)
    assert not decontam.is_contaminated(bench, set(), 13)


# ---------- KD loss ----------

def test_topk_kl_zero_when_student_matches_teacher():
    torch.manual_seed(0)
    logits = torch.randn(5, 50)
    lp = torch.log_softmax(logits, -1)
    v, ix = lp.topk(10, -1)
    assert topk_kl(logits, ix, v).item() < 1e-5


def test_topk_kl_positive_and_masks_padded_vocab():
    torch.manual_seed(0)
    s = torch.randn(4, 30)
    ids = torch.randint(0, 30, (4, 5))
    ids[:, -1] = 99  # outside the student vocab: must be ignored, not crash
    lp = torch.log_softmax(torch.randn(4, 5), -1)
    kl = topk_kl(s, ids, lp)
    assert math.isfinite(kl.item()) and kl.item() >= 0


def test_kd_loss_mixes_ce_and_kl():
    torch.manual_seed(0)
    s = torch.randn(3, 20, requires_grad=True)
    tgt = torch.tensor([1, 2, 3])
    ids = torch.randint(0, 20, (3, 4))
    lp = torch.log_softmax(torch.randn(3, 4), -1)
    loss, parts = kd_loss(s, tgt, ids, lp, alpha=0.3)
    assert abs(loss.item() - (0.3 * parts["ce"] + 0.7 * parts["kl"])) < 1e-4
    loss.backward()
    assert s.grad is not None


# ---------- data ----------

def test_collator_aligns_teacher_rows_with_labels():
    feats = [
        {"input_ids": [5, 6, 7, 8], "labels": [-100, -100, 7, 8], "target_start": 2,
         "topk_ids": [[7, 1], [8, 1]], "topk_logprobs": [[-0.1, -2.0], [-0.2, -2.0]]},
        {"input_ids": [5, 9], "labels": [-100, 9], "target_start": 1,
         "topk_ids": [[9, 2]], "topk_logprobs": [[-0.3, -1.5]]},
    ]
    b = PadCollator(pad_id=0, k=2)(feats)
    assert b["input_ids"].shape == (2, 4)
    assert b["topk_ids"][0, 2, 0] == 7 and b["topk_ids"][0, 3, 0] == 8
    assert b["topk_ids"][1, 1, 0] == 9 and b["labels"][1, 2] == -100
    m = b["labels"] != -100
    assert (b["topk_ids"][m][:, 0] == b["labels"][m]).all()


def test_topk_shard_roundtrip(tmp_path):
    import numpy as np

    save_topk_shard(tmp_path / "shard_00000.npz", ["a", "b"], [2, 1], [[1, 2, 3], [4, 5]],
                    [np.array([[3, 9]]), np.array([[5, 8]])], [np.array([[-0.1, -3.0]]), np.array([[-0.2, -2.0]])])
    rows = load_topk_shards(tmp_path)
    assert [r["id"] for r in rows] == ["a", "b"]
    assert rows[0]["input_ids"] == [1, 2, 3] and rows[1]["target_start"] == 1
    assert rows[1]["topk_ids"].tolist() == [[5, 8]]


# ---------- config ----------

def test_config_and_overrides(tmp_path, monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.setenv("SVLM_ROOT", str(tmp_path))
    cfg = load_config(root / "configs" / "code_fast.yaml", parse_overrides(["data.fim.n_train=10", "version=0.0.1"]))
    assert cfg["data"]["fim"]["n_train"] == 10 and cfg["version"] == "0.0.1"
    run = Run(cfg)
    assert run.run_dir == tmp_path / "runs" / "msci-code-fast" / "0.0.1-t4"
    assert not run.is_done("prepare")
    run.mark_done("prepare")
    assert run.is_done("prepare")
    with pytest.raises(KeyError):
        run.step_dir("nope")
    assert deep_merge({"a": {"b": 1, "c": 2}}, {"a": {"b": 3}}) == {"a": {"b": 3, "c": 2}}


def test_step_selection():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipelines"))
    from run_local import select_steps

    assert select_steps(None, None) == STEPS
    assert select_steps("verify,prepare", None) == ["prepare", "verify"]
    assert select_steps(None, "evaluate") == ["evaluate", "quantise", "register"]

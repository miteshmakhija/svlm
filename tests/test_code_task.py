"""Unit tests for the `code` task (msci-code): chat formatting, sandbox, problem parsing, verify, gates.
CPU only, no model downloads."""
import ast
import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svlm import chat, sandbox  # noqa: E402
from svlm.config import Run  # noqa: E402
from svlm.data import encode_example, prompt_and_target  # noqa: E402
from svlm.tasks import code  # noqa: E402
from svlm.utils import write_jsonl  # noqa: E402


class CharTok:
    """Tokenizer stand-in: one id per character."""

    def __call__(self, text, add_special_tokens=False):
        return {"input_ids": [ord(c) for c in text]}


# ---------------------------------------------------------------- chat formatting and encoding

def test_chatml_format_and_training_split():
    msgs = [{"role": "user", "content": "add 1 and 2"}, {"role": "assistant", "content": "3"}]
    prompt, target = chat.split_for_training(msgs, system="sys")
    assert prompt == "<|im_start|>system\nsys<|im_end|>\n<|im_start|>user\nadd 1 and 2<|im_end|>\n<|im_start|>assistant\n"
    assert target == "3<|im_end|>\n"
    # a leading system message overrides the default
    assert chat.format_chat([{"role": "system", "content": "S"}, {"role": "user", "content": "u"}]).startswith(
        "<|im_start|>system\nS<|im_end|>")


def test_encode_masks_prompt_for_chat_and_fim_rows():
    tok = CharTok()
    c = {"kind": "chat", "id": "c", "messages": [{"role": "user", "content": "q"}, {"role": "assistant", "content": "ans"}]}
    f = encode_example(tok, c, "code", 10_000)
    prompt, target = prompt_and_target(c)
    assert f["labels"][: len(prompt)] == [-100] * len(prompt)
    assert "".join(chr(i) for i in f["labels"][len(prompt):]) == target == "ans<|im_end|>\n"
    fim = {"id": "f", "prefix": "a = ", "suffix": "\n", "target": "1"}
    g = encode_example(tok, fim, "code", 10_000)
    assert "".join(chr(i) for i in g["labels"] if i != -100) == "1<|endoftext|>"
    assert encode_example(tok, c, "code", 5) is None  # too long


def test_extract_code_and_trim_reply():
    reply = "Use a loop.\n```python\ndef f(x):\n    return x\n```\nand\n```py\ny = 1\n```"
    assert code.extract_code(reply) == "def f(x):\n    return x\n\ny = 1"
    assert chat.extract_code("no fences here") == "no fences here"
    assert chat.extract_code("```python\ndef f():\n    pass") == "def f():\n    pass"  # unterminated: hit max tokens
    assert chat.trim_reply("done<|im_end|>\n<|im_start|>user") == "done"


# ---------------------------------------------------------------------------------- sandbox

def test_sandbox_pass_fail_timeout_and_setup():
    assert sandbox.run_tests("def add(a, b):\n    return a + b", ["assert add(1, 2) == 3"]).passed
    r = sandbox.run_tests("def add(a, b):\n    return a - b", ["assert add(1, 2) == 3", "assert add(0, 0) == 0"])
    assert not r.passed and r.n_passed == 1
    assert "timeout" in sandbox.run_tests("while True:\n    pass", ["assert True"], timeout=2).error
    assert sandbox.run_tests("y = x + 1", ["assert y == 2"], setup="x = 1").passed
    assert not sandbox.run_tests("", ["assert True"]).passed
    assert [x.passed for x in sandbox.run_many([("a = 1", ["assert a == 1"], ""), ("a = 2", ["assert a == 1"], "")])] == [True, False]


# --------------------------------------------------------------------------- problem parsing

def test_parse_problem_accepts_json_and_fenced_json():
    p = {"task": "Write f.", "entry_point": "f", "solution": "def f():\n    return 1", "tests": ["assert f() == 1", "assert f() != 2", "x"]}
    got = code.parse_problem(json.dumps(p))
    assert got["tests"] == ["assert f() == 1", "assert f() != 2"]  # non-assert lines dropped
    assert code.parse_problem("Here you go:\n```json\n" + json.dumps(p) + "\n```")["entry_point"] == "f"
    assert code.parse_problem("not json") is None
    assert code.parse_problem(json.dumps({**p, "tests": ["assert f() == 1"]})) is None  # needs >= 2 tests


def test_functions_strip_docstring_and_keep_valid_code():
    src = '''class A:
    def method(self, x):
        """Return x doubled.

        Long explanation here.
        """
        y = x * 2
        return y

def tiny():
    return 1
'''
    fns = {f["name"]: f for f in code._functions(src, 3, 40)}
    assert "tiny" not in fns  # below min_lines
    m = fns["method"]
    assert m["code"].startswith("def method(self, x):")  # dedented
    assert '"""' not in m["nodoc"] and "y = x * 2" in m["nodoc"]
    ast.parse(m["code"])
    ast.parse(m["nodoc"])


# ------------------------------------------------------------------------------------ verify

def _cfg(tmp_path):
    return {"name": "t", "version": "0", "tier": "x", "task": "code", "student": {}, "teacher": {"model": "t"},
            "paths": {"root": str(tmp_path)},
            "data": {"code": {"min_agree": 2, "test_timeout_s": 5, "max_explain_words": 250}}}


def test_verify_keeps_only_tested_answers(tmp_path):
    run = Run(_cfg(tmp_path))
    prep, gen = run.step_dir("prepare"), run.step_dir("teacher_generate")
    good = "```python\ndef sq(x):\n    return x * x\n```"
    bad = "```python\ndef sq(x):\n    return x + x\n```"
    seed = {"kind": "chat", "type": "synth_seed", "project": "p", "path": "a.py", "code": "def f(): pass"}
    write_jsonl(prep / "train.jsonl", [{"id": "fim1", "prefix": "a=", "suffix": "", "middle": "1", "span_type": "line"}])
    write_jsonl(prep / "chat_train.jsonl", [
        {"id": "m1", "kind": "chat", "type": "mbpp", "task": "Square x.", "tests": ["assert sq(3) == 9"], "setup": "", "split": "train"},
        {"id": "m2", "kind": "chat", "type": "mbpp", "task": "Square x.", "tests": ["assert sq(3) == 9"], "setup": "", "split": "train"},
        {**seed, "id": "s1", "split": "train"},   # reference ok, 2 of 4 agree -> kept
        {**seed, "id": "s2", "split": "train"},   # reference fails its own tests -> dropped
        {**seed, "id": "s3", "split": "train"},   # only 1 of 4 agree -> dropped
        {"id": "e1", "kind": "chat", "type": "explain", "code": "def f(): pass", "split": "train"},
        {"id": "d1", "kind": "chat", "type": "docstring", "split": "train",
         "messages": [{"role": "user", "content": "doc"}, {"role": "assistant", "content": "x"}]},
    ])
    write_jsonl(prep / "chat_heldout.jsonl", [
        {"id": "mh", "kind": "chat", "type": "mbpp", "task": "Square x.", "tests": ["assert sq(2) == 4"], "setup": "", "split": "heldout"},
        {**seed, "id": "sh", "split": "heldout"}])
    prob = lambda sol: json.dumps({"task": "Square x.", "entry_point": "sq", "solution": sol,  # noqa: E731
                                    "tests": ["assert sq(3) == 9", "assert sq(0) == 0"]})
    write_jsonl(gen / "problems" / "shard_00000.jsonl", [
        {"id": "s1", "samples": [prob("def sq(x):\n    return x * x")]},
        {"id": "s2", "samples": [prob("def sq(x):\n    return 0")]},
        {"id": "s3", "samples": [prob("def sq(x):\n    return x * x")]},
        {"id": "sh", "samples": [prob("def sq(x):\n    return x * x")]}])
    write_jsonl(gen / "solutions_train" / "shard_00000.jsonl", [
        {"id": "m1", "samples": [bad, good + "\nExtra words to make it longer.", good, bad]},
        {"id": "m2", "samples": [bad, bad, bad, bad]},
        {"id": "s1", "samples": [good, bad, good, bad]},
        {"id": "s2", "samples": [good, good, good, good]},
        {"id": "s3", "samples": [good, bad, bad, bad]}])
    write_jsonl(gen / "solutions_heldout" / "shard_00000.jsonl", [{"id": "mh", "samples": [good]}, {"id": "sh", "samples": [bad]}])
    write_jsonl(gen / "explain" / "shard_00000.jsonl", [{"id": "e1", "samples": ["short"]}])

    rep = code.verify(run)
    f = rep["teacher_train_filter"]
    assert f["mbpp"] == {"kept": 1, "no_solution_passes": 1}
    assert f["synth"] == {"kept": 1, "reference_fails_own_tests": 1, "too_few_agree": 1}
    assert f["explain"] == {"kept": 0, "length": 1}
    assert rep["sft_mix"] == {"fim": 1, "mbpp": 1, "synth": 1, "docstring": 1}
    sft = {r["id"]: r for r in map(json.loads, (run.step_dir("verify") / "sft.jsonl").read_text().splitlines())}
    assert sft["m1"]["messages"][-1]["content"] == good  # shortest passing answer
    assert sft["fim1"]["target"] == "1" and sft["fim1"]["kind"] == "fim"
    # held-out: MBPP + the synth problem whose reference passes; teacher solved MBPP, failed synth
    assert rep["chat_eval"] == {"n": 2, "mbpp": 1, "synth": 1}
    assert rep["teacher_heldout"]["chat"]["pass@1"] == 0.5


# ------------------------------------------------------------------------------------- gates

def _variant(es, parse, p1):
    return {"fim": {"edit_similarity": es, "parse_rate": parse}, "chat": {"pass@1": p1}}


def test_code_gates():
    cfg = {"gates": {"fim_reference_model": "msci-code-fast", "max_fim_drop_vs_reference": 0.02, "min_parse_rate": 0.8,
                     "chat_vs_teacher": 0.85, "max_chat_drop_vs_base": 0.02}}
    m = {"teacher": {"chat": {"pass@1": 0.6}}, "base": _variant(0.50, 0.7, 0.50), "final": _variant(0.60, 0.9, 0.52),
         "fim_reference": {"status": "ok", "version": "0.1.0-t4", "edit_similarity": 0.608}}
    g = code.apply_gates(cfg, m)
    assert g["verdict"] == "pass", g
    m["fim_reference"] = {"status": "different held-out spans"}
    g = code.apply_gates(cfg, m)
    assert g["verdict"] == "fail" and not g["checks"]["fim_vs_reference"]["pass"]  # never passes silently
    m["fim_reference"] = {"status": "ok", "version": "x", "edit_similarity": 0.608}
    m["final"] = _variant(0.60, 0.9, 0.45)  # 0.45 / 0.6 = 0.75 < 0.85 and below base - 0.02
    g = code.apply_gates(cfg, m)
    assert not g["checks"]["chat_vs_teacher"]["pass"] and not g["checks"]["chat_vs_base"]["pass"]


@pytest.mark.parametrize("rows,expected", [([], None), ([{"type": "a", "passed": 1.0}, {"type": "b", "passed": 0.0}], 0.5)])
def test_pass_summary(rows, expected):
    assert code.pass_summary(rows)["pass@1"] == expected

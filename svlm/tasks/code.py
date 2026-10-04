"""Task `code`: one coder for inline completion (FIM) and chat (msci-code).

Training data, mixed in one SFT set:
  fim        spans cut from the pinned repositories; the target is the ORIGINAL code (no teacher)
  mbpp       MBPP train/prompt problems; teacher writes n solutions, kept if they pass MBPP's own tests
  synth      problems the teacher writes from real repository functions (OSS-Instruct style), with a
             reference solution and asserts; the teacher then solves each problem n times and a problem
             is kept only if the reference passes its tests AND >= min_agree independent solutions do
             (guards against wrong teacher tests)
  docstring  "add a docstring" for repository functions; the target is the ORIGINAL docstring (no teacher)
  explain    teacher explains a repository function; kept if the length rules pass

Held-out (never trained on):
  heldout.jsonl       FIM spans from held-out files, built exactly like msci-code-fast's, so the two
                      models are scored on the same spans (checked by file hash before comparing)
  chat_heldout.jsonl  MBPP validation (ground-truth tests) + synth problems from held-out files

Evaluation reports FIM quality, chat pass@1 (tests run in svlm.sandbox) and latency for the
untrained base, SFT and final student, the teacher's chat pass@1 as the ceiling, and applies gates.
"""
from __future__ import annotations

import ast
import json
import random
import re
import textwrap
from pathlib import Path

from .. import decontam, sandbox
from ..chat import extract_code, format_chat, trim_reply
from ..config import Run
from ..utils import log, read_json, read_jsonl, run_sharded, sha256_file, stable_id, write_json, write_jsonl

TEACHER_SYSTEM = "You are an expert Python programmer."

PROBLEM_PROMPT = """Below is a function from the open-source project {project} (file `{path}`).

```python
{code}
```

Write ONE new, self-contained Python programming task inspired by what this function does: the same
kind of data handling or algorithm, but not a copy, and with no dependency on {project}. The solution
may use only the Python standard library, numpy and pandas.

Reply with a single JSON object and nothing else:
{{"task": "<what to implement, 2-6 sentences, naming the function and its parameters>",
 "entry_point": "<function name>",
 "solution": "<complete reference implementation, including imports>",
 "tests": ["assert ...", "assert ...", "assert ..."]}}

Give 3 to 6 independent one-line assert statements that check exact results (round floats before
comparing; compare DataFrames with .equals or by converting to lists)."""

SOLVE_SUFFIX = "\n\nYour code should pass this test:\n{test}"
SOLVE_SYSTEM = ("You are an expert Python programmer. Reply with one or two sentences on the approach, then the "
                "complete solution in a single ```python block, including imports. Do not include tests or example usage.")
EXPLAIN_PROMPT = ("Explain what this function does, for a colleague reviewing the code. Use 3-6 sentences or a short "
                  "bullet list; cover the inputs, the result and any edge cases.\n\n```python\n{code}\n```")
DOCSTRING_PROMPT = "Add a docstring to this function. Reply with the complete function in a ```python block.\n\n```python\n{code}\n```"


# ------------------------------------------------------------------------------------------ prepare


def _functions(content: str, min_lines: int, max_lines: int) -> list[dict]:
    """Functions and methods of a file as dedented source, with and without their docstring."""
    try:
        tree = ast.parse(content)
    except SyntaxError:
        return []
    lines = content.splitlines(keepends=True)
    out = []
    for fn in ast.walk(tree):
        if not isinstance(fn, (ast.FunctionDef, ast.AsyncFunctionDef)) or fn.name.startswith("__"):
            continue
        start = (fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno) - 1
        n = fn.end_lineno - start
        if not (min_lines <= n <= max_lines):
            continue
        src = textwrap.dedent("".join(lines[start:fn.end_lineno])).rstrip() + "\n"
        doc = ast.get_docstring(fn, clean=False)
        nodoc = None
        if doc is not None and len(fn.body) > 1:
            d0, d1 = fn.body[0].lineno - 1, fn.body[0].end_lineno
            nodoc = textwrap.dedent("".join(lines[start:d0] + lines[d1:fn.end_lineno])).rstrip() + "\n"
        out.append({"name": fn.name, "code": src, "nodoc": nodoc, "doc_chars": len(doc or "")})
    return out


def _mbpp(split: str) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("google-research-datasets/mbpp", "full", split=split)
    return [{"id": stable_id("mbpp", r["task_id"]), "kind": "chat", "type": "mbpp", "task": r["text"],
             "tests": list(r["test_list"]), "setup": r.get("test_setup_code") or "", "entry_point": None} for r in ds]


def solve_prompt(task: str, tests: list[str]) -> str:
    return task.strip() + SOLVE_SUFFIX.format(test=tests[0])


def prepare(run: Run) -> dict:
    from ..components.prepare_inputs import collect_sources, decontam_index, fim_samples, split_files

    cfg = run.cfg
    d, c = cfg["data"], cfg["data"]["code"]
    seed = cfg.get("seed", 42)
    out = run.step_dir("prepare")
    files, commits = collect_sources(run)
    held_files, train_files = split_files(files, d["fim"], seed)
    index, n = decontam_index(d)

    # FIM: identical to msci-code-fast's prepare, then a random subset of train for the mix
    fim_train, fim_held, drop_t, drop_h = fim_samples(train_files, held_files, d["fim"], seed, index, n)
    rng = random.Random(seed)
    fim_train = rng.sample(fim_train, min(c["fim_train"], len(fim_train)))

    def clean(rows, text):
        keep = [r for r in rows if not decontam.is_contaminated(text(r), index, n)]
        return keep, len(rows) - len(keep)

    # repository functions -> synth seeds, docstring and explain tasks (train files / held-out files)
    def fn_rows(fs, split):
        rows = []
        for f in fs:
            project = f["repo"].rstrip("/").split("/")[-1]
            for fn in _functions(f["content"], c.get("min_fn_lines", 5), c.get("max_fn_lines", 40)):
                rows.append({**fn, "project": project, "path": f["path"], "file_id": f["file_id"], "split": split})
        rng.shuffle(rows)
        return rows

    tr_fns, he_fns = fn_rows(train_files, "train"), fn_rows(held_files, "heldout")
    tr_fns, drop_fn = clean(tr_fns, lambda r: r["code"])

    def seed_row(fn, split):
        return {"id": stable_id(fn["file_id"], fn["name"], "synth"), "kind": "chat", "type": "synth_seed", "split": split,
                "project": fn["project"], "path": fn["path"], "code": fn["code"]}

    used: set[str] = set()

    def take(fns, k, pred):
        got = []
        for fn in fns:
            key = fn["file_id"] + fn["name"]
            if len(got) >= k:
                break
            if key not in used and pred(fn):
                used.add(key)
                got.append(fn)
        return got

    synth_tr = [seed_row(fn, "train") for fn in take(tr_fns, c["synth_train"], lambda f: True)]
    synth_he = [seed_row(fn, "heldout") for fn in take(he_fns, c["synth_heldout"], lambda f: True)]
    max_doc = c.get("max_docstring_chars", 1200)
    docs = [{"id": stable_id(fn["file_id"], fn["name"], "doc"), "kind": "chat", "type": "docstring", "split": "train",
             "messages": [{"role": "user", "content": DOCSTRING_PROMPT.format(code=fn["nodoc"].rstrip())},
                          {"role": "assistant", "content": f"```python\n{fn['code'].rstrip()}\n```"}]}
            for fn in take(tr_fns, c["docstring"], lambda f: f["nodoc"] and 40 <= f["doc_chars"] <= max_doc)]
    explain = [{"id": stable_id(fn["file_id"], fn["name"], "explain"), "kind": "chat", "type": "explain", "split": "train",
                "code": fn["code"]} for fn in take(tr_fns, c["explain"], lambda f: True)]

    mbpp_tr, mbpp_he = [], []
    if c.get("mbpp", True):
        mbpp_tr = [{**r, "split": "train"} for r in _mbpp("train") + _mbpp("prompt")][: c.get("mbpp_max") or None]
        mbpp_he = [{**r, "split": "heldout"} for r in _mbpp("validation")]  # MBPP test stays untouched (MBPP+)
        mbpp_tr, drop_m = clean(mbpp_tr, lambda r: r["task"] + "\n" + "\n".join(r["tests"]))
    else:
        drop_m = 0

    chat_train = mbpp_tr + synth_tr + docs + explain
    chat_held = mbpp_he + synth_he
    write_jsonl(out / "files.jsonl", files)
    write_jsonl(out / "train.jsonl", [{**r, "split": "train"} for r in fim_train])
    write_jsonl(out / "heldout.jsonl", [{**r, "split": "heldout"} for r in fim_held])
    write_jsonl(out / "chat_train.jsonl", chat_train)
    write_jsonl(out / "chat_heldout.jsonl", chat_held)

    def count(rows):
        out_: dict = {}
        for r in rows:
            out_[r["type"]] = out_.get(r["type"], 0) + 1
        return out_

    manifest = {
        "sources": commits,
        "n_files": len(files), "n_train_files": len(train_files), "n_heldout_files": len(held_files),
        "n_train": len(fim_train) + len(chat_train), "n_heldout": len(fim_held) + len(chat_held),
        "fim": {"train": len(fim_train), "heldout": len(fim_held)},
        "chat_train": count(chat_train), "chat_heldout": count(chat_held),
        "decontam_dropped": {"fim_train": drop_t, "fim_heldout": drop_h, "functions": drop_fn, "mbpp": drop_m,
                             "ngram": n, "index_size": len(index)},
        "sha256": {k: sha256_file(out / k) for k in ("train.jsonl", "heldout.jsonl", "chat_train.jsonl", "chat_heldout.jsonl")},
    }
    write_json(out / "manifest.json", manifest)
    log.info("prepare(code): fim %d/%d, chat train %s, chat held-out %s", len(fim_train), len(fim_held),
             manifest["chat_train"], manifest["chat_heldout"])
    return manifest


# ------------------------------------------------------------------------------- teacher_generate


def parse_problem(raw: str) -> dict | None:
    """The teacher's JSON problem, or None if it is unusable."""
    text = raw.strip()
    m = re.search(r"```(?:json)?\s*\n(.*?)```", text, re.S)
    if m:
        text = m.group(1)
    i, j = text.find("{"), text.rfind("}")
    if i == -1 or j <= i:
        return None
    try:
        p = json.loads(text[i:j + 1])
    except json.JSONDecodeError:
        return None
    tests = [t.strip() for t in p.get("tests", []) if isinstance(t, str) and t.strip().startswith("assert")]
    if not (isinstance(p.get("task"), str) and isinstance(p.get("solution"), str) and len(tests) >= 2):
        return None
    sol = extract_code(p["solution"]) if "```" in p["solution"] else p["solution"]
    return {"task": p["task"].strip(), "entry_point": str(p.get("entry_point") or ""), "solution": sol, "tests": tests[:6]}


def _problems(gen_dir: Path, seeds: list[dict]) -> list[dict]:
    raw = {}
    for p in sorted((gen_dir / "problems").glob("shard_*.jsonl")):
        raw.update({r["id"]: r["samples"][0] for r in read_jsonl(p)})
    out = []
    for s in seeds:
        p = parse_problem(raw.get(s["id"], ""))
        if p:
            out.append({**s, **p, "type": "synth"})
    return out


def teacher_generate(run: Run) -> dict:
    from ..teacher import ChatTeacher

    cfg = run.cfg
    g = cfg["teacher"]["gen"]
    prep, out = run.step_dir("prepare"), run.step_dir("teacher_generate")
    train, held = read_jsonl(prep / "chat_train.jsonl"), read_jsonl(prep / "chat_heldout.jsonl")
    state: dict = {}

    def teacher():
        if "t" not in state:
            state["t"] = ChatTeacher(cfg)
        return state["t"]

    def stage(name, rows, conv, n, temperature, max_tokens):
        def work(shard):
            replies = teacher().chat([conv(r) for r in shard], n=n, temperature=temperature if n == 1 else max(temperature, 0.5),
                                     max_tokens=max_tokens)
            return [{"id": r["id"], "samples": s} for r, s in zip(shard, replies)]
        return run_sharded(rows, out / name, work, shard_size=g.get("shard_size", 256), desc=f"teacher {name}")

    sys_msg = lambda s: [{"role": "system", "content": s}]  # noqa: E731
    seeds = [r for r in train + held if r["type"] == "synth_seed"]
    try:
        stage("problems", seeds, lambda r: sys_msg(TEACHER_SYSTEM) + [{"role": "user", "content": PROBLEM_PROMPT.format(
            project=r["project"], path=r["path"], code=r["code"].rstrip())}], 1, g.get("problem_temperature", 0.7), 1024)
        probs = _problems(out, seeds)
        solve_conv = lambda r: sys_msg(SOLVE_SYSTEM) + [{"role": "user", "content": solve_prompt(r["task"], r["tests"])}]  # noqa: E731
        tr_jobs = [r for r in train if r["type"] == "mbpp"] + [p for p in probs if p["split"] == "train"]
        he_jobs = [r for r in held if r["type"] == "mbpp"] + [p for p in probs if p["split"] == "heldout"]
        stage("solutions_train", tr_jobs, solve_conv, g.get("n_solutions", 4), g.get("temperature", 0.7), g.get("max_new_tokens", 768))
        stage("solutions_heldout", he_jobs, solve_conv, 1, 0.0, g.get("max_new_tokens", 768))  # greedy: the ceiling
        stage("explain", [r for r in train if r["type"] == "explain"], lambda r: sys_msg(TEACHER_SYSTEM) + [
            {"role": "user", "content": EXPLAIN_PROMPT.format(code=r["code"].rstrip())}], 1, 0.2, 400)
    finally:
        if "t" in state:
            state["t"].close()
    info = {"teacher": cfg["teacher"]["model"], "backend": cfg["teacher"].get("backend", "vllm"), "seeds": len(seeds),
            "problems_parsed": len(probs), "solve_train": len(tr_jobs), "solve_heldout": len(he_jobs)}
    write_json(out / "summary.json", info)
    log.info("teacher_generate(code): %s", info)
    return info


# ------------------------------------------------------------------------------------------ verify


def _samples(d: Path) -> dict[str, list[str]]:
    out: dict[str, list[str]] = {}
    for p in sorted(d.glob("shard_*.jsonl")):
        out.update({r["id"]: r["samples"] for r in read_jsonl(p)})
    return out


def _chat_row(r: dict, reply: str) -> dict:
    return {"id": r["id"], "kind": "chat", "type": r["type"],
            "messages": [{"role": "user", "content": solve_prompt(r["task"], r["tests"])}, {"role": "assistant", "content": reply.strip()}]}


def verify(run: Run) -> dict:
    cfg = run.cfg
    c = cfg["data"]["code"]
    prep, gen, out = run.step_dir("prepare"), run.step_dir("teacher_generate"), run.step_dir("verify")
    train, held = read_jsonl(prep / "chat_train.jsonl"), read_jsonl(prep / "chat_heldout.jsonl")
    timeout = c.get("test_timeout_s", 10)
    min_agree = c.get("min_agree", 2)
    sol_tr, sol_he = _samples(gen / "solutions_train"), _samples(gen / "solutions_heldout")
    probs = {p["id"]: p for p in _problems(gen, [r for r in train + held if r["type"] == "synth_seed"])}
    rep: dict = {t: {"kept": 0} for t in ("mbpp", "synth", "explain", "docstring")}

    def bump(t, k):
        rep[t][k] = rep[t].get(k, 0) + 1

    # run every candidate of every train problem in one parallel batch
    tr_probs = [r for r in train if r["type"] == "mbpp"] + [p for p in probs.values() if p["split"] == "train"]
    jobs, owners = [], []
    for r in tr_probs:
        if r["type"] == "synth":  # reference first: wrong teacher tests are caught here
            jobs.append((r["solution"], r["tests"], ""))
            owners.append((r["id"], -1))
        for k, s in enumerate(sol_tr.get(r["id"], [])):
            jobs.append((extract_code(s), r["tests"], r.get("setup", "")))
            owners.append((r["id"], k))
    log.info("verify(code): running %d candidate programs in the sandbox", len(jobs))
    results: dict[str, dict[int, bool]] = {}
    for (rid, k), res in zip(owners, sandbox.run_many(jobs, timeout=timeout)):
        results.setdefault(rid, {})[k] = res.passed

    sft = [{**r, "target": r["middle"], "kind": "fim", "source": "original"} for r in read_jsonl(prep / "train.jsonl")]
    for r in tr_probs:
        res, cands = results.get(r["id"], {}), sol_tr.get(r["id"], [])
        if not cands:
            bump(r["type"], "no_teacher_output")
            continue
        if r["type"] == "synth" and not res.get(-1, False):
            bump("synth", "reference_fails_own_tests")
            continue
        passing = [k for k in range(len(cands)) if res.get(k)]
        if not passing:
            bump(r["type"], "no_solution_passes")
            continue
        if r["type"] == "synth" and len(passing) < min_agree:
            bump("synth", "too_few_agree")
            continue
        best = min(passing, key=lambda k: len(cands[k]))  # shortest passing answer: less verbose student
        sft.append({**_chat_row(r, cands[best]), "source": "teacher"})
        bump(r["type"], "kept")

    exp = _samples(gen / "explain")
    for r in (r for r in train if r["type"] == "explain"):
        reply = (exp.get(r["id"]) or [""])[0].strip()
        words = len(reply.split())
        if not 25 <= words <= c.get("max_explain_words", 250):
            bump("explain", "length")
            continue
        sft.append({"id": r["id"], "kind": "chat", "type": "explain", "source": "teacher", "messages": [
            {"role": "user", "content": EXPLAIN_PROMPT.format(code=r["code"].rstrip())}, {"role": "assistant", "content": reply}]})
        bump("explain", "kept")
    for r in (r for r in train if r["type"] == "docstring"):
        sft.append({**r, "source": "original"})
        bump("docstring", "kept")

    # held-out chat problems with trustworthy tests: MBPP's own, or synth whose reference passes
    he_probs = [r for r in held if r["type"] == "mbpp"] + [p for p in probs.values() if p["split"] == "heldout"]
    ref_ok = sandbox.run_many([(p["solution"], p["tests"], "") for p in he_probs if p["type"] == "synth"], timeout=timeout)
    ok_ids = {p["id"] for p, res in zip([p for p in he_probs if p["type"] == "synth"], ref_ok) if res.passed}
    eval_set = [{k: p[k] for k in ("id", "type", "task", "tests", "setup") if k in p}
                for p in he_probs if p["type"] == "mbpp" or p["id"] in ok_ids]
    t_res = sandbox.run_many([(extract_code((sol_he.get(p["id"]) or [""])[0]), p["tests"], p.get("setup", "")) for p in eval_set],
                             timeout=timeout)
    teacher_rows = [{"id": p["id"], "type": p["type"], "passed": float(r.passed)} for p, r in zip(eval_set, t_res)]

    write_jsonl(out / "sft.jsonl", sft)
    write_jsonl(out / "chat_eval.jsonl", eval_set)
    write_jsonl(out / "teacher_heldout_scores.jsonl", teacher_rows)
    kinds: dict = {}
    for r in sft:
        key = r.get("type") or r.get("kind", "fim")
        kinds[key] = kinds.get(key, 0) + 1
    report = {
        "teacher_train_filter": rep,
        "n_sft": len(sft),
        "sft_mix": kinds,
        "n_sft_teacher": sum(1 for r in sft if r["source"] == "teacher"),
        "chat_eval": {"n": len(eval_set), "mbpp": sum(p["type"] == "mbpp" for p in eval_set),
                      "synth": sum(p["type"] == "synth" for p in eval_set)},
        "teacher_heldout": {"chat": pass_summary(teacher_rows)},
        "sha256": {"sft.jsonl": sha256_file(out / "sft.jsonl"), "chat_eval.jsonl": sha256_file(out / "chat_eval.jsonl")},
    }
    write_json(out / "filter_report.json", report)
    log.info("verify(code): %s", {k: report[k] for k in ("sft_mix", "chat_eval", "teacher_heldout")})
    return report


def pass_summary(rows: list[dict]) -> dict:
    def rate(rs):
        return round(sum(r["passed"] for r in rs) / len(rs), 4) if rs else None
    return {"n": len(rows), "pass@1": rate(rows), "by_type": {t: rate([r for r in rows if r["type"] == t])
                                                               for t in sorted({r["type"] for r in rows})}}


# ---------------------------------------------------------------------------------------- evaluate


def chat_variant(cfg: dict, adapter: Path | None, problems: list[dict], out: Path, name: str) -> dict:
    """Greedy answers from one student variant, scored by running the tests. Cached like the FIM eval."""
    from ..components.evaluate import _cache_key
    from ..modeling import free_gpu, generate_batch, load_student_for_inference
    from ..utils import chunks

    e = cfg["eval"]
    pred_path, meta_path = out / f"chat_predictions_{name}.jsonl", out / f"chat_predictions_{name}.meta.json"
    key = {**_cache_key({"max_new_tokens": e.get("chat_max_new_tokens", 512)}, adapter, len(problems))}
    if pred_path.exists() and meta_path.exists() and read_json(meta_path).get("key") == key:
        return pass_summary(read_jsonl(pred_path))
    log.info("evaluate %s: chat answers for %d problems", name, len(problems))
    model, tok = load_student_for_inference(cfg, adapter)
    prompts = [format_chat([{"role": "user", "content": solve_prompt(p["task"], p["tests"])}]) for p in problems]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    replies = [""] * len(prompts)
    for b in chunks(order, e.get("chat_batch_size", 16)):
        for i, r in zip(b, generate_batch(model, tok, [prompts[i] for i in b], e.get("chat_max_new_tokens", 512))):
            replies[i] = trim_reply(r)
    model = None
    free_gpu()
    res = sandbox.run_many([(extract_code(r), p["tests"], p.get("setup", "")) for r, p in zip(replies, problems)],
                           timeout=cfg["data"]["code"].get("test_timeout_s", 10))
    rows = [{"id": p["id"], "type": p["type"], "passed": float(x.passed), "error": x.error, "reply": r}
            for p, r, x in zip(problems, replies, res)]
    write_jsonl(pred_path, rows)
    write_json(meta_path, {"key": key})
    s = pass_summary(rows)
    log.info("evaluate %s: chat pass@1=%s %s", name, s["pass@1"], s["by_type"])
    return s


def fim_reference(run: Run, model: str) -> dict:
    """Live metrics of the reference FIM model, if it was scored on the same held-out spans."""
    import yaml

    cat_path = run.model_store / "catalogue.yaml"
    if not cat_path.exists():
        return {"status": f"no catalogue at {cat_path}"}
    live = (yaml.safe_load(cat_path.read_text()).get("models", {}).get(model) or {}).get("live")
    if not live:
        return {"status": f"{model} has no live version"}
    d = run.model_store / model / live
    ref_sha = read_json(d / "lineage.json")["data_sha256"].get("heldout.jsonl")
    ours = read_json(run.step_dir("prepare") / "manifest.json")["sha256"]["heldout.jsonl"]
    if ref_sha != ours:
        return {"status": f"{model}:{live} was scored on different held-out spans (hash mismatch)", "version": live}
    final = read_json(d / "metrics.json")["final"]
    return {"status": "ok", "version": live, "edit_similarity": final["edit_similarity"], "exact_match": final["exact_match"],
            "parse_rate": final["parse_rate"]}


def apply_gates(cfg: dict, m: dict) -> dict:
    g = cfg.get("gates", {})
    f, b, t = m["final"], m["base"], m["teacher"]
    checks: dict = {}
    ref = m.get("fim_reference", {})
    if g.get("fim_reference_model"):
        thr = g.get("max_fim_drop_vs_reference", 0.02)
        if ref.get("status") == "ok":
            v = f["fim"]["edit_similarity"]
            checks["fim_vs_reference"] = {"value": v, "threshold": round(ref["edit_similarity"] - thr, 4),
                                          "reference": f"{g['fim_reference_model']}:{ref['version']}",
                                          "pass": v >= ref["edit_similarity"] - thr}
        else:  # never pass silently when the comparison cannot be made
            checks["fim_vs_reference"] = {"value": ref.get("status", "missing"), "threshold": "-", "pass": False}
    checks["fim_beats_base"] = {"value": f["fim"]["edit_similarity"], "base": b["fim"]["edit_similarity"],
                                "pass": f["fim"]["edit_similarity"] > b["fim"]["edit_similarity"]}
    p_thr = g.get("min_parse_rate", 0.80)
    checks["fim_parse_rate"] = {"value": f["fim"]["parse_rate"], "threshold": p_thr, "pass": f["fim"]["parse_rate"] >= p_thr}
    tp, fp, bp = t["chat"]["pass@1"], f["chat"]["pass@1"], b["chat"]["pass@1"]
    q = g.get("chat_vs_teacher", 0.85)
    checks["chat_vs_teacher"] = {"value": round(fp / tp, 3) if tp else "teacher score missing", "threshold": q,
                                 "pass": bool(tp) and fp / tp >= q}
    drop = g.get("max_chat_drop_vs_base", 0.02)
    checks["chat_vs_base"] = {"value": fp, "threshold": round(bp - drop, 4), "pass": fp >= bp - drop}
    ttft = f["fim"].get("latency", {}).get("ttft_ms_p50")
    if ttft is not None and "max_ttft_ms_gpu" in g:
        checks["ttft_gpu"] = {"value": ttft, "threshold": g["max_ttft_ms_gpu"], "pass": ttft <= g["max_ttft_ms_gpu"]}
    return {"checks": checks, "verdict": "pass" if all(c["pass"] for c in checks.values()) else "fail"}


def evaluate(run: Run) -> dict:
    from ..components.evaluate import eval_variant
    from ..utils import iter_jsonl

    cfg = run.cfg
    out = run.step_dir("evaluate")
    held = read_jsonl(run.step_dir("prepare") / "heldout.jsonl")
    files = {f["file_id"]: f["content"] for f in iter_jsonl(run.step_dir("prepare") / "files.jsonl")}
    problems = read_jsonl(run.step_dir("verify") / "chat_eval.jsonl")
    sft = run.step_dir("train_sft") / "adapter"
    kd = run.step_dir("train_kd") / "adapter"
    final_adapter = kd if kd.exists() else sft

    def variant(adapter, name):
        return {"fim": eval_variant(cfg, adapter, held, files, out, name), "chat": chat_variant(cfg, adapter, problems, out, name)}

    report = read_json(run.step_dir("verify") / "filter_report.json")
    final = variant(final_adapter, "final")  # "final" also measures latency
    metrics = {
        "teacher": report["teacher_heldout"],
        "base": variant(None, "base"),
        "sft": variant(sft, "sft") if final_adapter != sft else final,  # KD off: final is the SFT adapter
        "final": final,
        "final_adapter": str(final_adapter),
    }
    if cfg.get("gates", {}).get("fim_reference_model"):
        metrics["fim_reference"] = fim_reference(run, cfg["gates"]["fim_reference_model"])
    metrics["gates"] = apply_gates(cfg, metrics)
    write_json(out / "metrics.json", metrics)
    log.info("evaluate(code): verdict=%s fim_es=%s chat_pass@1=%s (teacher %s, base %s)", metrics["gates"]["verdict"],
             metrics["final"]["fim"]["edit_similarity"], metrics["final"]["chat"]["pass@1"],
             metrics["teacher"]["chat"]["pass@1"], metrics["base"]["chat"]["pass@1"])
    return metrics

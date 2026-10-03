"""Fill-in-the-middle (FIM) sample construction, prompt formatting and scoring.

Qwen2.5-Coder FIM format:
    <|fim_prefix|>{code before cursor}<|fim_suffix|>{code after cursor}<|fim_middle|>{completion}

Samples are stored with character offsets into the original file so a completion can always be
spliced back into the *full* file and checked with `ast.parse`, even though the prompt only
carries a truncated prefix and suffix.
"""
from __future__ import annotations

import ast
import difflib
import random
from dataclasses import dataclass

FIM_PREFIX = "<|fim_prefix|>"
FIM_SUFFIX = "<|fim_suffix|>"
FIM_MIDDLE = "<|fim_middle|>"
FIM_PAD = "<|fim_pad|>"
EOS = "<|endoftext|>"
STOP_STRINGS = [EOS, FIM_PREFIX, FIM_SUFFIX, FIM_MIDDLE, FIM_PAD, "<|file_sep|>", "<|repo_name|>"]

SPAN_TYPES = ("line", "block", "function_body")


def fim_prompt(prefix: str, suffix: str) -> str:
    return f"{FIM_PREFIX}{prefix}{FIM_SUFFIX}{suffix}{FIM_MIDDLE}"


@dataclass
class Span:
    span_type: str
    start: int  # char offset, inclusive
    end: int    # char offset, exclusive


class _Lines:
    """Line-offset table for a file."""

    def __init__(self, text: str):
        self.text = text
        self.starts = [0]
        for i, ch in enumerate(text):
            if ch == "\n":
                self.starts.append(i + 1)
        self.n = len(self.starts)

    def start(self, lineno: int) -> int:  # 1-based
        return self.starts[lineno - 1]

    def end(self, lineno: int) -> int:  # offset just after the line's newline
        return self.starts[lineno] if lineno < self.n else len(self.text)

    def line(self, lineno: int) -> str:
        return self.text[self.start(lineno) : self.end(lineno)]


def _functions(tree: ast.AST) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    return [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]


def _body_first_line(fn) -> int | None:
    """First line of the body, skipping a docstring. None for docstring-only functions."""
    body = fn.body
    if body and isinstance(body[0], ast.Expr) and isinstance(getattr(body[0], "value", None), ast.Constant) \
            and isinstance(body[0].value.value, str):
        body = body[1:]
    return body[0].lineno if body else None


def candidate_spans(text: str, max_middle_lines: int, rng: random.Random) -> dict[str, list[Span]]:
    """All usable spans in one file, grouped by span type. Returns {} if the file does not parse."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return {}
    L = _Lines(text)
    out: dict[str, list[Span]] = {t: [] for t in SPAN_TYPES}

    # line: cursor somewhere inside a code line, complete to end of line
    for ln in range(1, L.n + 1):
        raw = L.line(ln).rstrip("\n")
        stripped = raw.strip()
        if len(stripped) < 8 or stripped.startswith("#"):
            continue
        indent = len(raw) - len(raw.lstrip())
        cut = indent + rng.randint(1, max(1, len(stripped) - 3))
        out["line"].append(Span("line", L.start(ln) + cut, L.start(ln) + len(raw)))

    for fn in _functions(tree):
        first = _body_first_line(fn)
        last = getattr(fn, "end_lineno", None)
        if first is None or last is None or last < first:
            continue
        n_body = last - first + 1
        # function_body: the whole body of a short function
        if 2 <= n_body <= max_middle_lines:
            out["function_body"].append(Span("function_body", L.start(first), L.end(last)))
        # block: 2..max lines from inside the body
        if n_body >= 3:
            size = rng.randint(2, min(max_middle_lines, n_body - 1))
            s = rng.randint(first, last - size + 1)
            out["block"].append(Span("block", L.start(s), L.end(s + size - 1)))
    return out


def truncate_context(text: str, span: Span, max_prefix_chars: int, max_suffix_chars: int) -> tuple[str, str]:
    """Prefix/suffix around a span, trimmed to budget on line boundaries."""
    prefix = text[: span.start]
    if len(prefix) > max_prefix_chars:
        cut = len(prefix) - max_prefix_chars
        nl = prefix.find("\n", cut)
        prefix = prefix[nl + 1 :] if nl != -1 else prefix[cut:]
    suffix = text[span.end :]
    if len(suffix) > max_suffix_chars:
        nl = suffix.rfind("\n", 0, max_suffix_chars)
        suffix = suffix[: nl + 1] if nl != -1 else suffix[:max_suffix_chars]
    return prefix, suffix


def build_samples(
    files: list[dict],
    n_total: int,
    span_mix: dict[str, float],
    max_middle_lines: int,
    max_prefix_chars: int,
    max_suffix_chars: int,
    seed: int = 42,
    max_middle_chars: int = 1200,
) -> list[dict]:
    """Draw `n_total` FIM samples across files following `span_mix`."""
    rng = random.Random(seed)
    pools: dict[str, list[tuple[dict, Span]]] = {t: [] for t in SPAN_TYPES}
    for f in files:
        for t, spans in candidate_spans(f["content"], max_middle_lines, rng).items():
            for sp in spans:
                middle = f["content"][sp.start : sp.end]
                if middle.strip() and len(middle) <= max_middle_chars:
                    pools[t].append((f, sp))
    weights = {t: span_mix.get(t, 0.0) for t in SPAN_TYPES}
    total_w = sum(weights.values()) or 1.0
    samples: list[dict] = []
    seen: set[tuple[str, int, int]] = set()
    for t in SPAN_TYPES:
        want = round(n_total * weights[t] / total_w)
        pool = pools[t]
        rng.shuffle(pool)
        for f, sp in pool:
            if want <= 0:
                break
            key = (f["file_id"], sp.start, sp.end)
            if key in seen:
                continue
            seen.add(key)
            prefix, suffix = truncate_context(f["content"], sp, max_prefix_chars, max_suffix_chars)
            samples.append(
                {
                    "id": f"{f['file_id']}-{sp.start}-{sp.end}",
                    "file_id": f["file_id"],
                    "repo": f["repo"],
                    "path": f["path"],
                    "license": f["license"],
                    "span_type": t,
                    "start": sp.start,
                    "end": sp.end,
                    "prefix": prefix,
                    "suffix": suffix,
                    "middle": f["content"][sp.start : sp.end],
                }
            )
            want -= 1
    rng.shuffle(samples)
    return samples


# ---------- post-processing and scoring ----------

def trim_completion(text: str, span_type: str, suffix: str) -> str:
    """Cut a raw generation down to the part that fills the hole."""
    for s in STOP_STRINGS:
        i = text.find(s)
        if i != -1:
            text = text[:i]
    if span_type == "line":
        return text.split("\n", 1)[0]
    # if the model ran on into the suffix, stop where the suffix's first real line starts
    first_suffix_line = next((ln for ln in suffix.splitlines() if ln.strip()), None)
    if first_suffix_line:
        i = text.find(first_suffix_line)
        if i > 0:
            text = text[:i]
    return text


def _norm(s: str) -> str:
    return "\n".join(line.rstrip() for line in s.strip("\n").splitlines()).strip()


def exact_match(pred: str, ref: str) -> bool:
    return _norm(pred) == _norm(ref)


def edit_similarity(pred: str, ref: str) -> float:
    a, b = _norm(pred), _norm(ref)
    if not a and not b:
        return 1.0
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio()


def parses_in_file(content: str, start: int, end: int, completion: str) -> bool:
    try:
        ast.parse(content[:start] + completion + content[end:])
        return True
    except (SyntaxError, ValueError):
        return False


def score(pred: str, sample: dict, content: str | None = None) -> dict:
    out = {
        "exact_match": float(exact_match(pred, sample["middle"])),
        "edit_similarity": edit_similarity(pred, sample["middle"]),
    }
    if content is not None:
        out["parses"] = float(parses_in_file(content, sample["start"], sample["end"], pred))
    return out

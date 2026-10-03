"""Config loading and run-directory layout.

A config is a YAML file (see configs/). Every component receives the same `Run` object,
which knows where each step reads and writes. Layout under the root:

    <root>/runs/<name>/<version>-<tier>/<step>/...      step outputs
    <root>/runs/<name>/<version>-<tier>/<step>/_SUCCESS  written when a step completes
    <root>/<model_store>/<name>/<version>-<tier>/        released artefacts (register step)
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

STEPS = [
    "prepare",
    "teacher_generate",
    "verify",
    "teacher_logits",
    "train_sft",
    "train_kd",
    "evaluate",
    "quantise",
    "register",
]


def load_config(path: str | os.PathLike, overrides: dict | None = None) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        cfg = deep_merge(cfg, overrides)
    root = os.environ.get("SVLM_ROOT")
    if root:
        cfg.setdefault("paths", {})["root"] = root
    _validate(cfg)
    return cfg


def deep_merge(base: dict, extra: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in extra.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_overrides(pairs: list[str]) -> dict:
    """Turn ["data.fim.n_train=200", "teacher.model=Qwen/x"] into a nested dict."""
    out: dict = {}
    for pair in pairs:
        key, _, raw = pair.partition("=")
        value: Any = yaml.safe_load(raw) if raw != "" else ""
        node = out
        parts = key.split(".")
        for p in parts[:-1]:
            node = node.setdefault(p, {})
        node[parts[-1]] = value
    return out


def _validate(cfg: dict) -> None:
    for key in ("name", "version", "tier", "task", "student", "teacher", "paths"):
        if key not in cfg:
            raise ValueError(f"config is missing required key '{key}'")


def config_hash(cfg: dict) -> str:
    blob = json.dumps(cfg, sort_keys=True, default=str).encode()
    return hashlib.sha256(blob).hexdigest()[:16]


@dataclass
class Run:
    cfg: dict

    @property
    def root(self) -> Path:
        return Path(self.cfg["paths"]["root"]).expanduser()

    @property
    def tag(self) -> str:
        return f"{self.cfg['version']}-{self.cfg['tier']}"

    @property
    def run_dir(self) -> Path:
        return self.root / "runs" / self.cfg["name"] / self.tag

    def step_dir(self, step: str) -> Path:
        if step not in STEPS:
            raise KeyError(step)
        d = self.run_dir / step
        d.mkdir(parents=True, exist_ok=True)
        return d

    @property
    def model_store(self) -> Path:
        p = Path(self.cfg["paths"].get("model_store", "model_store")).expanduser()
        return p if p.is_absolute() else self.root / p

    def is_done(self, step: str) -> bool:
        return (self.run_dir / step / "_SUCCESS").exists()

    def mark_done(self, step: str, info: dict | None = None) -> None:
        payload = {"step": step, "config_hash": config_hash(self.cfg), **(info or {})}
        (self.step_dir(step) / "_SUCCESS").write_text(json.dumps(payload, indent=2, default=str))

    def snapshot_config(self) -> None:
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "config.resolved.yaml").write_text(yaml.safe_dump(self.cfg, sort_keys=False))

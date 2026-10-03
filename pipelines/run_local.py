"""Run pipeline steps locally (Colab, a workstation, or one Kubeflow pod per step).

Examples
    python pipelines/run_local.py --config configs/code_fast.yaml                    # all steps
    python pipelines/run_local.py --config configs/code_fast.yaml --steps prepare,teacher_generate
    python pipelines/run_local.py --config configs/code_fast.yaml --from train_sft
    python pipelines/run_local.py --config configs/code_fast.yaml --set data.fim.n_train=300 --set version=0.0.1
    python pipelines/run_local.py --config configs/code_fast.yaml --status

Completed steps are skipped (a `_SUCCESS` marker per step); use --force to redo them.
"""
from __future__ import annotations

import argparse
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from svlm import components  # noqa: E402
from svlm.config import STEPS, Run, load_config, parse_overrides  # noqa: E402
from svlm.utils import log  # noqa: E402


def select_steps(steps: str | None, start: str | None) -> list[str]:
    if steps and steps != "all":
        chosen = [s.strip() for s in steps.split(",") if s.strip()]
        bad = [s for s in chosen if s not in STEPS]
        if bad:
            raise SystemExit(f"unknown step(s): {bad}. Valid: {STEPS}")
        return [s for s in STEPS if s in chosen]
    if start:
        if start not in STEPS:
            raise SystemExit(f"unknown step {start}. Valid: {STEPS}")
        return STEPS[STEPS.index(start):]
    return list(STEPS)


def status(run: Run) -> None:
    print(f"{run.cfg['name']}:{run.tag}  run dir: {run.run_dir}")
    for s in STEPS:
        print(f"  {'done ' if run.is_done(s) else 'todo '} {s}")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--steps", help="comma-separated steps, or 'all'")
    ap.add_argument("--from", dest="start", help="run from this step to the end")
    ap.add_argument("--set", action="append", default=[], help="override a config value, e.g. data.fim.n_train=300")
    ap.add_argument("--force", action="store_true", help="re-run steps even if marked done")
    ap.add_argument("--status", action="store_true")
    a = ap.parse_args(argv)

    cfg = load_config(a.config, parse_overrides(a.set))
    run = Run(cfg)
    if a.status:
        status(run)
        return 0
    run.snapshot_config()
    for step in select_steps(a.steps, a.start):
        if run.is_done(step) and not a.force:
            log.info("skip %s (done)", step)
            continue
        log.info("=== %s ===", step)
        t0 = time.time()
        try:
            info = components.get(step).run(run)
        except Exception:
            traceback.print_exc()
            log.error("step %s failed; fix and re-run (finished steps are kept)", step)
            return 1
        run.mark_done(step, {"seconds": round(time.time() - t0, 1), "info": info})
    status(run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

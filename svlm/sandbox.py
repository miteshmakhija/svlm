"""Run model-written Python against assert-style tests in a separate, short-lived process.

Each run gets a fresh temporary directory, an isolated interpreter (`python -I`), a wall-clock
timeout and, on Linux, CPU-time / memory / file-size limits. This is a guard against accidents
(infinite loops, huge allocations, stray files), not a security boundary: the code can still
use the network on Colab. Only public, teacher- or student-written code is run here.
"""
from __future__ import annotations

import os
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

_RUNNER = r'''
import sys
src = open(sys.argv[1], encoding="utf-8").read()
tests = open(sys.argv[2], encoding="utf-8").read().split("\n#--TEST--\n")
g = {"__name__": "__sandbox__"}
exec(compile(src, "solution.py", "exec"), g)
ok = 0
for t in tests:
    if not t.strip():
        continue
    try:
        exec(compile(t, "test.py", "exec"), g)
        ok += 1
    except AssertionError:
        print("FAIL", t.strip().splitlines()[0][:200])
    except Exception as e:
        print("ERROR", type(e).__name__, str(e)[:200], "|", t.strip().splitlines()[0][:200])
print("PASSED", ok)
'''


@dataclass
class Result:
    passed: bool
    n_passed: int
    n_tests: int
    error: str = ""


def _limits():  # pragma: no cover - Linux only, runs in the child
    import resource

    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))
    resource.setrlimit(resource.RLIMIT_AS, (2 << 30, 2 << 30))
    resource.setrlimit(resource.RLIMIT_FSIZE, (10 << 20, 10 << 20))
    os.setsid()


def run_tests(code: str, tests: list[str], setup: str = "", timeout: float = 10.0) -> Result:
    """Execute `setup` + `code`, then each test; passed only if every test passes."""
    tests = [t for t in tests if t.strip()]
    if not code.strip() or not tests:
        return Result(False, 0, len(tests), "no code" if not code.strip() else "no tests")
    with tempfile.TemporaryDirectory(prefix="svlm-sbx-") as d:
        sol, tst, run = Path(d, "solution.py"), Path(d, "tests.txt"), Path(d, "runner.py")
        sol.write_text((setup + "\n" if setup else "") + code, encoding="utf-8")
        tst.write_text("\n#--TEST--\n".join(tests), encoding="utf-8")
        run.write_text(_RUNNER, encoding="utf-8")
        env = {"PATH": os.environ.get("PATH", ""), "PYTHONHASHSEED": "0", "PYTHONIOENCODING": "utf-8",
               "SYSTEMROOT": os.environ.get("SYSTEMROOT", "")}  # SYSTEMROOT: Windows needs it to start Python
        try:
            p = subprocess.run([sys.executable, "-I", str(run), str(sol), str(tst)], cwd=d, env=env,
                               capture_output=True, text=True, timeout=timeout,
                               preexec_fn=_limits if os.name == "posix" else None)
        except subprocess.TimeoutExpired:
            return Result(False, 0, len(tests), f"timeout after {timeout:.0f}s")
    out = p.stdout.strip().splitlines()
    n_ok = int(out[-1].split()[1]) if out and out[-1].startswith("PASSED") else 0
    err = "" if n_ok == len(tests) else (p.stderr.strip().splitlines()[-1][:300] if p.stderr.strip() else
                                          next((ln for ln in out if ln.startswith(("FAIL", "ERROR"))), "failed"))
    return Result(n_ok == len(tests), n_ok, len(tests), err)


def run_many(jobs: list[tuple[str, list[str], str]], workers: int | None = None, timeout: float = 10.0) -> list[Result]:
    """run_tests for many (code, tests, setup) jobs in parallel processes, results in input order."""
    workers = workers or max(2, (os.cpu_count() or 2))
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda j: run_tests(j[0], j[1], j[2], timeout), jobs))

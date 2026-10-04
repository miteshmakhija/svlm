"""Task plugins. The pipeline components handle `task: fim` themselves (milestone M1) and hand
every other task to its module here, which provides the task-specific steps:

    prepare(run) -> dict            build train / held-out inputs
    teacher_generate(run) -> dict   teacher outputs (sharded, resumable)
    verify(run) -> dict             checkers -> sft.jsonl + filter_report.json
    evaluate(run) -> dict           student variants, teacher ceiling, release gates -> metrics.json

Training, logit distillation, quantisation and registration are shared by all tasks.
"""
from importlib import import_module

TASKS = {"code": "svlm.tasks.code"}


def get(task: str):
    if task not in TASKS:
        raise KeyError(f"unknown task {task!r}; known: fim, {', '.join(TASKS)}")
    return import_module(TASKS[task])

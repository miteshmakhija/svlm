"""Pipeline components. Each module exposes `run(run: Run) -> dict` and is idempotent.

Step order (see svlm.config.STEPS):
    prepare -> teacher_generate -> verify -> teacher_logits -> train_sft -> train_kd
            -> evaluate -> quantise -> register
"""
from importlib import import_module

MODULES = {
    "prepare": "svlm.components.prepare_inputs",
    "teacher_generate": "svlm.components.teacher_generate",
    "verify": "svlm.components.verify_filter",
    "teacher_logits": "svlm.components.teacher_logits",
    "train_sft": "svlm.components.train_sft",
    "train_kd": "svlm.components.train_kd",
    "evaluate": "svlm.components.evaluate",
    "quantise": "svlm.components.quantise",
    "register": "svlm.components.register",
}


def get(step: str):
    return import_module(MODULES[step])

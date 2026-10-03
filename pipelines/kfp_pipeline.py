"""Kubeflow Pipelines (KFP v2) definition — cluster tier.

The same components that run on Colab run here, one pod per step, sharing a PVC that plays
the role of Google Drive. Each pod runs:

    python pipelines/run_local.py --config <cfg> --steps <step> --set paths.root=/data/svlm

Compile:
    pip install kfp kfp-kubernetes
    python pipelines/kfp_pipeline.py --config configs/code_fast.yaml --image <registry>/svlm:0.1.0 --out code_fast.yaml

Then upload the YAML in the Kubeflow UI (or with kfp.Client().create_run_from_pipeline_package).
"""

import argparse

from kfp import compiler, dsl, kubernetes

STEPS = ["prepare", "teacher_generate", "verify", "teacher_logits", "train_sft", "train_kd", "evaluate", "quantise", "register"]
GPU_STEPS = {"teacher_generate", "teacher_logits", "train_sft", "train_kd", "evaluate", "quantise"}
PVC_NAME = "svlm-data"
MOUNT = "/data/svlm"


def build(config_path: str, image: str, gpu_type: str = "nvidia.com/gpu"):
    def make_step(step: str):
        @dsl.container_component
        def _step(config: str, root: str):
            return dsl.ContainerSpec(
                image=image,
                command=["python", "pipelines/run_local.py"],
                args=["--config", config, "--steps", step, "--set", dsl.ConcatPlaceholder(["paths.root=", root])],
            )

        _step.component_spec.name = f"svlm-{step.replace('_', '-')}"
        return _step

    ops = {s: make_step(s) for s in STEPS}

    @dsl.pipeline(name="svlm-distillation", description="Teacher -> student distillation for one catalogue model")
    def pipeline(config: str = config_path, root: str = MOUNT):
        prev = None
        for s in STEPS:
            task = ops[s](config=config, root=root)
            task.set_caching_options(False)  # steps manage their own resume markers on the PVC
            kubernetes.mount_pvc(task, pvc_name=PVC_NAME, mount_path=MOUNT)
            if s in GPU_STEPS:
                task.set_accelerator_type(gpu_type).set_accelerator_limit(1)
                task.set_memory_limit("48G").set_cpu_limit("8")
            else:
                task.set_memory_limit("16G").set_cpu_limit("4")
            if prev is not None:
                task.after(prev)
            prev = task

    return pipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gpu-type", default="nvidia.com/gpu")
    a = ap.parse_args()
    compiler.Compiler().compile(build(a.config, a.image, a.gpu_type), a.out)
    print(f"compiled {a.out}")


if __name__ == "__main__":
    main()

from dataclasses import dataclass
import os
from pathlib import Path
import argparse

@dataclass
class Case:
    model: str
    n_gpus: int

CASES = [
    Case(model="Qwen/Qwen2.5-VL-3B-Instruct", n_gpus=1),
    Case(model="Qwen/Qwen2.5-VL-7B-Instruct", n_gpus=1),
    Case(model="Qwen/Qwen2.5-VL-32B-Instruct", n_gpus=2),
    Case(model="Qwen/Qwen2.5-VL-72B-Instruct", n_gpus=4),
]

BASE_PORT = 9000


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    return parser.parse_args()


tgt_dir = Path("slurm_scripts")
os.makedirs(tgt_dir, exist_ok=True)

# Clear any existing .sbatch files
for file in tgt_dir.glob("*.sbatch"):
    file.unlink()

template: str = Path("template.sbatch").read_text()
for i, case in enumerate(CASES):
    port = BASE_PORT + (i * 100)  # Give enough space between job ports
    jobname = f"{i}_{case.model.replace('/', '_')}"
    filled = template.format(
        JOB_NAME=jobname,
        MODEL_ID=case.model,
        VLLM_PORT=port,
        N_GPUS=case.n_gpus,
    )

    script_path = tgt_dir / f"{jobname}.sbatch"
    script_path.write_text(filled)
    os.chmod(script_path, 0o755)
    print(f"Created sbatch file: '{script_path}'")

    args = parse_args()
    if args.launch:
        os.system(f"sbatch {script_path}")
        print(f"Launched array job: '{script_path}'")

import os
import subprocess
from pathlib import Path
import argparse
from dataclasses import dataclass
from typing import List
import logging

logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen2.5-VL-72B-Instruct"
N_GPUS = 4
RUNS_PER_MODALITY = 5

MODALITIES_FOR_ABLATION = ["only_video", "transcribed"]
FIXED_TEMP_FOR_ABLATION = 0.0

MODALITY_FOR_TEMP_SWEEP = "transcribed"
TEMPERATURES_FOR_SWEEP = [0.2, 0.5, 0.7, 1.0]
RUNS_PER_TEMP = 5

@dataclass
class ExperimentRun:
    model: str
    n_gpus: int
    modality: str
    temperature: float
    run_index: int

def generate_experiment_sequences() -> List[ExperimentRun]:
    exp_sequences = []

    for modality in MODALITIES_FOR_ABLATION:
        for i in range(1, RUNS_PER_MODALITY + 1):
            exp_sequences.append(ExperimentRun(
                model=MODEL_ID,
                n_gpus=N_GPUS,
                modality=modality,
                temperature=FIXED_TEMP_FOR_ABLATION,
                run_index=i
            ))
    logger.info(f"Added {len(exp_sequences)} modality ablation runs.")

    
    temp_sweep_runs = []
    for temp in TEMPERATURES_FOR_SWEEP:
        for i in range(1, RUNS_PER_TEMP + 1):
            temp_sweep_runs.append(ExperimentRun(
                model=MODEL_ID,
                n_gpus=N_GPUS,
                modality=MODALITY_FOR_TEMP_SWEEP,
                temperature=temp,
                run_index=i
            ))
    exp_sequences.extend(temp_sweep_runs)
    logger.info(f"Added {len(temp_sweep_runs)} temperature sweep runs.")

    logger.info(f"Total runs: {len(exp_sequences)}")
    return exp_sequences

def submit_job(script_path: Path, dependency_id: str = None) -> str:
    command = ["sbatch"]
    if dependency_id:
        command.append(f"--dependency=afterok:{dependency_id}")
    command.append(str(script_path))
    
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    job_id = result.stdout.strip().split()[-1]
    logger.info(f"Submitted job {job_id} for {script_path.name} (dependency: {dependency_id or 'None'})")
    return job_id

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--launch", action="store_true")
    args = parser.parse_args()

    experiments = generate_experiment_sequences()

    tgt_dir = Path("slurm_scripts")
    tgt_dir.mkdir(exist_ok=True)
    for file in tgt_dir.glob("*.sbatch"):
        file.unlink() # delete any existing .sbatch files

    script_dir = Path(__file__).parent
    template_path = script_dir / "qwen_template.sbatch"
    template: str = template_path.read_text()
    script_paths = []

    logger.info("Generating .sbatch files...")
    for i, run in enumerate(experiments):
        job_name = f"{i:02d}_{run.modality}_t{run.temperature}_r{run.run_index}"
        port = 9000 + i * 100 # sequential runs but still for safety

        filled = template.format(
            JOB_NAME=job_name,
            MODEL_ID=run.model,
            VLLM_PORT=port,
            N_GPUS=run.n_gpus,
            INPUT_STRATEGY=run.modality,
            TEMPERATURE=run.temperature
        )

        script_path = tgt_dir / f"{job_name}.sbatch"
        script_path.write_text(filled)
        os.chmod(script_path, 0o755)
        script_paths.append(script_path)
        logger.info(f"Created: {script_path.name}")

    if args.launch:
        logger.info("Launching job chain...")
        last_job_id = None
        for path in script_paths:
            last_job_id = submit_job(path, dependency_id=last_job_id)
        logger.info(f"Submitted job chain. The final job is {last_job_id}.")
        logger.info("Monitor progress with: squeue -u $USER")

if __name__ == "__main__":
    main()
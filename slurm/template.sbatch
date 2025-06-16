#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4
#SBATCH --gres=gpu:{N_GPUS}
#SBATCH --array=0
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/toxicainment/sbatch/%u/%x/%a/stdout.txt
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/toxicainment/sbatch/%u/%x/%a/stderr.txt
#SBATCH --job-name {JOB_NAME}
#SBATCH --time 5:00:00

# Set container name based on job name and array task ID to avoid conflicts
CONTAINER_NAME="toxicainment_{JOB_NAME}_${{SLURM_ARRAY_TASK_ID}}"

# Calculate port based on base port and array task ID to avoid conflicts
VLLM_PORT=$(({VLLM_PORT} + $SLURM_ARRAY_TASK_ID))

# Delete any existing container with this name
echo "Removing any existing container named $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true

# Create new container
echo "Creating new container $CONTAINER_NAME..."
enroot create --name $CONTAINER_NAME /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/tomasruiz/my_custom_pt.sqsh

enroot start --rw --root \
    --mount /dss/dsshome1/0D/di38bec/code:/workspace/code \
    --mount /dss/dsshome1/0D/di38bec/datasets:/workspace/datasets \
    --mount "$DSS_HOME:$DSS_HOME" \
    --env DSS_HOME=$DSS_HOME \
    --env HF_TOKEN=$HF_TOKEN \
    --env HF_HOME=$HF_HOME \
    --env BUGSNAG_API_KEY=$BUGSNAG_API_KEY \
    $CONTAINER_NAME bash -c "source /root/ptvenv/bin/activate && python /workspace/code/simple-inference-benchmark/run_benchmark.py \
    --model_id={MODEL_ID} \
    --n_examples=300 \
    --use_vllm=True \
    --vllm_remote_call_concurrency=32 \
    --vllm_start_server=True \
    --vllm_port $VLLM_PORT \
    --vllm_allowed_local_media_path /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/ \
    --vllm_tensor_parallel_size {N_GPUS} \
    --tgt_file /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/toxicainment/sbatch/di38bec/{JOB_NAME}/all-responses.jsonl"

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true
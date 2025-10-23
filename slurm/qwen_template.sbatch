#!/bin/bash
#SBATCH --partition lrz-hgx-h100-94x4,lrz-dgx-a100-80x8,lrz-hgx-a100-80x4
#SBATCH --gres=gpu:{N_GPUS}
#SBATCH --job-name {JOB_NAME}
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/toxicainment/sbatch/%u/%x/stdout.txt
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/toxicainment/sbatch/%u/%x/stderr.txt
#SBATCH --job-name {JOB_NAME}
#SBATCH --time 3:00:00

# Set container name based on job name
CONTAINER_NAME="toxicainment_{JOB_NAME}"

# Delete any existing container with this name
echo "Removing any existing container named $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true

# Create new container
echo "Creating new container $CONTAINER_NAME..."
enroot create --name $CONTAINER_NAME /dss/dsshome1/02/ra79vom2/agustoslu+pytorch_with_deps.sqsh

enroot start --rw --root \
    --mount /dss/dsshome1/02/ra79vom2/code:/workspace/code \
    --mount "$DSS_HOME:$DSS_HOME" \
    --env DSS_HOME=$DSS_HOME \
    --env HF_TOKEN=$HF_TOKEN \
    --env HF_HOME=$HF_HOME \
    $CONTAINER_NAME bash -c "source /root/code/tenv/bin/activate && python /workspace/code/simple-inference-benchmark/run_benchmark.py \
    --model_id={MODEL_ID} \
    --restart=False \
    --n_examples=2 \
    --use_vllm=True \
    --vllm_remote_call_concurrency=32 \
    --temperature={TEMPERATURE} \
    --vllm_start_server=True \
    --vllm_port {VLLM_PORT} \
    --vllm_allowed_local_media_path /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/ \
    --vllm_tensor_parallel_size {N_GPUS} \
    --input_strategy={INPUT_STRATEGY} \
    --tgt_file /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/toxicainment/sbatch/ra79vom2/{JOB_NAME}/all-responses.jsonl"
     

# Clean up: remove the container after job completion
echo "Cleaning up: removing container $CONTAINER_NAME..."
enroot remove --force $CONTAINER_NAME || true
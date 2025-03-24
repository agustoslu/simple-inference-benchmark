#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -o /dss/dsshome1/02/ra79vom2/toxicainment/simple-inference-benchmark/enroot_test.out
#SBATCH -e /dss/dsshome1/02/ra79vom2/toxicainment/simple-inference-benchmark/enroot_test.err
#SBATCH --job-name test-job
#SBATCH --time 04:00:00

enroot create --name my_custom_pt /dss/dsshome1/02/ra79vom2/agustoslu+pytorch_with_deps.sqsh
enroot start --rw --root \
    --mount /dss/dsshome1/02/ra79vom2/code:/workspace/code \
    --mount /dss/dsshome1/02/ra79vom2/datasets:/workspace/datasets \
    --mount "$DSS_HOME:$DSS_HOME" \
    --env DSS_HOME=$DSS_HOME \
    --env HF_TOKEN=$HF_TOKEN \
    --env HF_HOME=$HF_HOME \
    my_container /root/env/bin/python /workspace/toxicainment/simple-inference-benchmark/run_benchmark.py --model_id='Qwen/Qwen2.5-VL-7B-Instruct' --n_examples=300
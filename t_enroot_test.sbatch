#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/sbatch/%u/%x/enroot_test.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/sbatch/%u/%x/enroot_test.err
#SBATCH --job-name q2_5-3b-it
#SBATCH --time 04:00:00

enroot create --name my_container /dss/dsshome1/02/ra79vom2/agustoslu+pytorch_with_deps.sqsh
enroot start --rw --root \
    --mount /dss/dsshome1/02/ra79vom2/code:/workspace/code \
    --mount /dss/dsshome1/02/ra79vom2/datasets:/workspace/datasets \
    --mount "$DSS_HOME:$DSS_HOME" \
    --env DSS_HOME=$DSS_HOME \
    --env HF_TOKEN=$HF_TOKEN \
    --env HF_HOME=$HF_HOME \
    my_container /root/tenv/bin/python /workspace/code/simple-inference-benchmark/run_benchmark.py --model_id='Qwen/Qwen2.5-VL-3B-Instruct' --n_examples=300
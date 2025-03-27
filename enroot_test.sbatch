#!/bin/bash
#SBATCH -p lrz-hgx-h100-94x4
#SBATCH --gres=gpu:1
#SBATCH -o /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/sbatch/%u/%x/enroot_test.out
#SBATCH -e /dss/dssfs02/lwp-dss-0001/pn76je/pn76je-dss-0000/sbatch/%u/%x/enroot_test.err
#SBATCH --job-name g3-27b-it-04
#SBATCH --time 04:00:00

enroot create --name my_custom_pt /dss/dsshome1/0D/di38bec/my_custom_pt.sqsh
enroot start --rw --root \
    --mount /dss/dsshome1/0D/di38bec/code:/workspace/code \
    --mount /dss/dsshome1/0D/di38bec/datasets:/workspace/datasets \
    --mount "$DSS_HOME:$DSS_HOME" \
    --env DSS_HOME=$DSS_HOME \
    --env HF_TOKEN=$HF_TOKEN \
    --env HF_HOME=$HF_HOME \
    my_custom_pt /root/ptvenv/bin/python /workspace/code/simple-inference-benchmark/run_benchmark.py --model_id='google/gemma-3-27b-it' --n_examples=300
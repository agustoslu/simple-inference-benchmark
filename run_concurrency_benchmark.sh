#!/bin/bash

# Array of concurrency values to test
concurrency_values=(32 24 16 12 8 4 2 1)

# Loop through each concurrency value
for concurrency in "${concurrency_values[@]}"; do
    echo "Running benchmark with concurrency: $concurrency"
    
    # Run the benchmark with the current concurrency value
    python run_benchmark.py \
        --model_id=Qwen/Qwen2.5-VL-3B-Instruct \
        --use_vllm=True \
        --n_examples=100 \
        --vllm_remote_call_concurrency=$concurrency
    
    # Add a small delay between runs to ensure clean state
    sleep 2
done

echo "All benchmarks completed!" 
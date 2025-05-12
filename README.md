# Inference on ToxicAInment Data

Run the inference workload:

```bash
python run_benchmark.py --model_id="openbmb/MiniCPM-V-2_6" --n_examples=300

python run_benchmark.py --model_id="google/gemma-3-4b-it" --n_examples=300
```

Run a test experiment on 24 GB GPU and using vLLM

```bash
python run_benchmark.py --use_vllm=True --max_n_frames_per_video=10 --model_id=Qwen/Qwen2.5-VL-3B-Instruct --vllm_max_model_len=16384 --n_examples=10
```

Run the experiment in a slurm job

```bash
sbatch enroot_test.sbatch
```

## vLLM Batch Inference

Run a vLLM batch inference. See this example [documentation](https://github.com/vllm-project/vllm/blob/main/examples/offline_inference/openai/openai_batch.md):

First create a batch file:

```bash
python make_vlmm_batch.py --model_id=Qwen/Qwen2.5-VL-3B-Instruct --tgt_jsonl=vllm-batch/tomas/batch_input.jsonl --n_examples=10
```

Then cd to the directory and run the batch inference:

```bash
cd vllm-batch/tomas
python -m vllm.entrypoints.openai.run_batch -i batch_input.jsonl -o results.jsonl --model Qwen/Qwen2.5-VL-3B-Instruct --allowed-local-media-path=/home/
```

## vLLM Server

Run a vLLM server:

```bash
vllm serve Qwen/Qwen2.5-VL-3B-Instruct --max-model-len 32768 --max-seq-len-to-capture 32768 --dtype bfloat16 --allowed-local-media-path=/home/ --limit-mm-per-prompt "image=50,video=2"
```

Then send a single request:

```bash
curl http://localhost:8000/v1/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "Qwen/Qwen2.5-VL-3B-Instruct",
        "prompt": "San Francisco is a",
        "max_tokens": 7,
        "temperature": 0
    }'
```

Or run the experiment against the vLLM server:

```bash
python run_benchmark.py --model_id=Qwen/Qwen2.5-VL-3B-Instruct --use_vllm=True --n_examples=30 --vllm_remote_call_concurrency=8
```

# Install bench_lib
The `bench_lib` is a package containing utility code and evaluation code. By installing it you can import it everywhere (e.g. within jupyter notebook in `notebooks/`) without worrying about the path of files. 
```bash
pip install -e bench_lib
```

# Install llmlib
You want to have both codebases side by side for development purposes, so that you can introduce changes to `llmlib` and immediately test them in this project.
```bash
git clone https://github.com/tomasruizt/llm_app
cd llm_app/llmlib
pip install -e .
```

# Troubleshooting

#### Error: Gemma3 not found
```shell
ImportError: cannot import name 'Gemma3ForConditionalGeneration' from 'transformers'
```
This happend because the `transformers` version is too early and does not contain the Gemma3 model yet. Install this version:
```shell
pip install git+https://github.com/huggingface/transformers@v4.49.0-Gemma-3
```

#### RuntimeError: Cannot re-initialize CUDA in forked subprocess

Full error:

```shell
RuntimeError: Cannot re-initialize CUDA in forked subprocess. To use CUDA with multiprocessing, you must use the 'spawn' start method
```

Solution: Add this vLLM env var: `VLLM_WORKER_MULTIPROC_METHOD=spawn`

Full description in the [GitHub issue](https://github.com/vllm-project/vllm/issues/8893)

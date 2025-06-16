.PHONY: run-benchmark
.PHONY: vllm-server

run-benchmark:
	python run_benchmark.py \
		--model_id Qwen/Qwen2.5-VL-3B-Instruct \
		--n_examples 2 \
		--use_vllm True \
		--vllm_remote_call_concurrency 2


vllm-server:
	vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
		--max-model-len 32768 \
		--max-seq-len-to-capture 32768 \
		--dtype auto \
		--allowed-local-media-path=/home/ \
		--limit-mm-per-prompt "image=50,video=1" \
		--disable-log-requests \
		--port 8000 \
		--gpu-memory-utilization 0.8 \
		--disable-uvicorn-access-log \
		--enforce-eager

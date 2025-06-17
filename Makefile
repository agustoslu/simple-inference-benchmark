.PHONY: run-benchmark
.PHONY: vllm-server

run-benchmark:
	python run_benchmark.py \
		--model_id Qwen/Qwen2.5-VL-3B-Instruct \
		--n_examples 30 \
		--use_vllm True \
		--vllm_remote_call_concurrency 8 \
		--restart False \
		--vllm_start_server False \
		--dataset_dir /mnt/disk16tb/globus_shared/20250516_btw_3pct_subset


vllm-server:
	vllm serve Qwen/Qwen2.5-VL-3B-Instruct \
		--max-model-len 32768 \
		--max-seq-len-to-capture 32768 \
		--dtype auto \
		--allowed-local-media-path=/ \
		--limit-mm-per-prompt "image=50,video=1" \
		--disable-log-requests \
		--port 8000 \
		--host 127.0.0.1 \
		--disable-uvicorn-access-log \
		--gpu-memory-utilization 0.8 \

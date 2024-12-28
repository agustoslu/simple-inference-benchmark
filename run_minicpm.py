import os
import time
import random
import yaml
import cv2
import json
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer

# Data paths
DATASET_PATH = "/dss/dsshome1/02/ra79vom2/simple-inference-benchmark/dataset/FineVideo_20_Samples"
TEMP_VIDEO_DIR = "/dss/dsshome1/02/ra79vom2/simple-inference-benchmark/dataset/temp_videos"
LOG_FILE = "benchmark_log.txt"

os.makedirs(TEMP_VIDEO_DIR, exist_ok=True)

# Extracting parquet files
def extract_videos_from_parquet(dataset_path, temp_video_dir, num_videos, seed):
    video_paths = []
    parquet_files = [f for f in os.listdir(dataset_path) if f.endswith(".parquet")] 

    random.seed(seed)

    for parquet_file in tqdm(parquet_files, desc="Processing Parquet files"):
        parquet_path = os.path.join(dataset_path, parquet_file)
        df = pd.read_parquet(parquet_path)

        for index, row in df.iterrows():
            video_binary = row['mp4']
            temp_video_path = os.path.join(temp_video_dir, f"{parquet_file}_video_{index}.mp4")
            with open(temp_video_path, "wb") as f:
                f.write(video_binary)
            video_paths.append(temp_video_path)

    random_videos = random.sample(video_paths, num_videos)
    return random_videos

def process_video(video_path, seconds_per_frame, token_limit, num_samples, model, tokenizer):
    cap = cv2.VideoCapture(video_path)
    original_fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = max(1, int(original_fps * seconds_per_frame))  # Frames to skip
    frame_count = 0
    tokens_generated = 0
    queries = 0
    model_runtime = 0
    extra_runtime = 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", unit="frame")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_pil = Image.fromarray(frame_rgb)
            
            # Prompt for generation
            question = "Describe this video in detail"
            msgs = [{"role": "user", "content": [frame_pil, question]}]

            # Model inference
            start_model_time = time.time()
            
            try:
                for _ in range(num_samples):
                    res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, sampling=False, stream=False)
                    print(f"Generated Text for Frame {frame_count}: {res}")
            except Exception as e:
                print(f"Error processing frame {frame_count}: {e}")
                res = ""
            model_runtime += time.time() - start_model_time

            # Calculate time taken for additional operations
            start_extra_time = time.time()
            num_tokens = len(tokenizer.tokenize(res))
            tokens_generated += num_tokens
            queries += 1
            extra_runtime += time.time() - start_extra_time

        frame_count += 1
        progress_bar.update(1)

    cap.release()
    progress_bar.close()

    return tokens_generated, queries, model_runtime, extra_runtime


def benchmark_videos(video_paths, seconds_per_frame, token_limit, num_samples, hf_token, compile):
    model = AutoModel.from_pretrained(
        "openbmb/MiniCPM-V-2_6",
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        use_auth_token=hf_token
    ).eval().cuda()

    if compile:
        model = torch.compile(model)

    tokenizer = AutoTokenizer.from_pretrained(
        "openbmb/MiniCPM-V-2_6", trust_remote_code=True, use_auth_token=hf_token
    )

    results = []
    total_runtime = 0
    total_model_runtime = 0
    total_extra_runtime = 0
    total_queries = 0
    total_tokens = 0
    total_peak_memory_opencv = 0
    total_peak_memory_inference = 0
    global_peak_memory_allocated = 0 

    for video_path in tqdm(video_paths, desc="Benchmarking videos"):
        print(f"\nProcessing: {video_path}")
        #torch.cuda.reset_peak_memory_stats()
        start_time = time.time()
        
        initial_memory_allocated = torch.cuda.memory_allocated() / 1e9
        initial_memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{os.path.basename(video_path)}] Initial Memory - Allocated: {initial_memory_allocated:.3f} GB, Reserved: {initial_memory_reserved:.3f} GB")        

        tokens_generated, queries, model_runtime, extra_runtime = process_video(
            video_path, seconds_per_frame, token_limit, num_samples, model, tokenizer
        )
        peak_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
        peak_memory_reserved = torch.cuda.max_memory_reserved() / 1e9
        global_peak_memory_allocated = max(global_peak_memory_allocated, peak_memory_allocated)
        global_peak_memory_reserved = max(global_peak_memory_reserved, peak_memory_reserved)

        video_runtime = time.time() - start_time

        print(f"[{os.path.basename(video_path)}] Peak Memory - Allocated: {peak_memory_allocated:.3f} GB, Reserved: {peak_memory_reserved:.3f} GB")

        print(f"Finished {os.path.basename(video_path)}")
        print(f"  Total Runtime: {video_runtime:.2f}s")
        print(f"  Model Runtime: {model_runtime:.2f}s")
        print(f"  Extra Operations Runtime: {extra_runtime:.2f}s")
        print(f"  Tokens Generated: {tokens_generated}")
        print(f"  Queries Made: {queries}")
        
        total_runtime += video_runtime
        total_model_runtime += model_runtime
        total_extra_runtime += extra_runtime
        total_queries += queries
        total_tokens += tokens_generated

        results.append({
            "video": video_path,
            "runtime": video_runtime,
            "model_runtime": model_runtime,
            "extra_runtime": extra_runtime,
            "tokens": tokens_generated,
            "queries": queries,
            "peak_memory_allocated": peak_memory_allocated,
            "peak_memory_reserved": peak_memory_reserved,
        })

    qps = total_queries / total_model_runtime if total_model_runtime > 0 else 0
    tps = total_tokens / total_model_runtime if total_model_runtime > 0 else 0 
    tpq = total_tokens / total_queries if total_queries > 0 else 0 

    print("\nBenchmark Summary:")
    print(f"  Total Runtime: {total_runtime:.2f}s")
    print(f"  Queries per Second (QPS): {qps:.2f}")
    print(f"  Tokens per Second (TPS): {tps:.2f}")
    print(f"  Tokens per Query (TPQ): {tpq:.2f}")
    print(f"  Global Peak Memory Allocated: {global_peak_memory_allocated:.3f} GB")
    print(f"  Global Peak Memory Reserved: {global_peak_memory_reserved:.3f} GB")

    with open(LOG_FILE, "a") as log_file:
        log_file.write(f"Total Runtime: {total_runtime}\n")
        log_file.write(f"Model Runtime: {total_model_runtime}\n")
        log_file.write(f"Extra Runtime: {total_extra_runtime}\n")
        log_file.write(f"QPS: {qps}\n")
        log_file.write(f"TPS: {tps}\n")
        log_file.write(f"TPQ: {tpq}\n")
        log_file.write(f"Global Peak Memory Allocated: {global_peak_memory_allocated:.3f} GB\n")
        log_file.write(f"Global Peak Memory Reserved: {global_peak_memory_reserved:.3f} GB\n")
        log_file.write("\n")

    return results

if __name__ == "__main__":
    with open("config.yaml", "r") as config_file:
        config = yaml.safe_load(config_file)

    print("Extracting videos from Parquet files...")
    video_paths = extract_videos_from_parquet(
        DATASET_PATH, TEMP_VIDEO_DIR, num_videos=config["num_videos"], seed=config["seed"]
    )

    print("Benchmarking videos...")
    benchmark_results = benchmark_videos(
        video_paths,
        seconds_per_frame=config["fps_settings"],
        token_limit=config["token_settings"], 
        num_samples=config["num_samples"], 
        hf_token=config["hf_token"],
        compile=config["compile"]
    )

    for video_path in video_paths:
        os.remove(video_path)
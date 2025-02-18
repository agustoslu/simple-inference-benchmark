import os
from pathlib import Path
import time
from datetime import datetime
import random
import csv
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pyaml_env import parse_config
import duckdb
from decord import VideoReader, cpu
from download import DATASET_PATH, MP4_DATASET_PATH
from collections import defaultdict
from utils import get_posts

# Extract and sample videos
def sample_n_videos(n: int, seed: int):
    con = duckdb.connect()
    n_videos = con.sql(f"SELECT COUNT(*) FROM '{DATASET_PATH}/*.parquet'").fetchone()[0]
    print(f"Total videos: {n_videos}")
    random.seed(seed)
    random_ids = random.sample(range(n_videos), n)
    print(f"Randomly selected video IDs: {random_ids}")
    df = con.sql(f"""
                 SELECT id, mp4
                 FROM (SELECT mp4, ROW_NUMBER() OVER () AS id FROM '{DATASET_PATH}/*.parquet')
                 WHERE id IN ({', '.join([str(id) for id in random_ids])})
                 """).fetch_arrow_table()

    video_paths = []
    for idx, row in zip(df["id"], df["mp4"]):
        mp4_video_path = os.path.join(MP4_DATASET_PATH, f"video_{idx}.mp4")
        if not Path(mp4_video_path).exists():
            with open(mp4_video_path, "wb") as f:
                f.write(row.as_py())
            print(f"Saved video to {mp4_video_path}")
        video_paths.append(mp4_video_path)

    return video_paths

# Employ uniform sampling for frames
def uniform_sample(xs, n):
    gap = len(xs) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [xs[i] for i in idxs]

def fps_sample(xs, fps, total_range): 
    idxs = [i * fps for i in range(xs // fps)]
    return [total_range[i] for i in idxs]

def load_model(model_id: str, config: dict):
    """Loads model from huggingface"""
    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        token=config["hf_token"],
        device_map="cuda",
    ).eval()
    return model

def create_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)

def metadata_generator(meta_data):
    for i in meta_data:
        yield i

def process_video(video_path, token_limit, num_samples, model, tokenizer, meta_data):


    # Say hello to your function :)
    MAX_NUM_FRAMES = 64
    frame_indices = []
    
    # pass metadata, we get each instance inside benchmark_videos using generator
    author = meta_data["author_name"]
    video = meta_data["video_description"]


    try:
        vr = VideoReader(str(video_path), ctx=cpu(0))
        total = []
        total_frames = len(vr)
        for i in vr:
            total.append(i)
        
        #breakpoint()
        print(f"Total Frames: {total_frames}")
        fps = vr.get_avg_fps()
        print(f"FPS: {fps}")
        total_frames_int = int(total_frames)
        print(f"total transformed: {total_frames_int}")
        if total_frames <= 6000:
            frame_indices = fps_sample(int(total_frames), round(fps), range(total_frames))
            print(f"frame indices: {frame_indices}")
            print(f"total frames passed(fps): {len(frame_indices)}")

        elif total_frames > 6000:
            total_fps = total[:6000]
            total_uni = total[6000:]
            # total uni starts from zero since it's newly created even tough frames after 4000 are added
            frame_indices = fps_sample(int(len(total_fps)), round(fps), range(len(total_fps)))
            frame_indices_uni = uniform_sample(range(len(total_uni)), MAX_NUM_FRAMES)
            print(f"frame indices_fps: {frame_indices}")
            print(f"total frames passed(fps): {len(frame_indices)}")
            print(f"frame indices_uni: {frame_indices_uni}")
            print(f"total frames passed(uni): {len(frame_indices_uni)}")
        
        frames = vr.get_batch(frame_indices).asnumpy()
        frames = [Image.fromarray(frame.astype("uint8")) for frame in frames]
    except Exception as e:
        raise ValueError(f"Error processing video: {e}") from e

    tokens_generated = 0
    num_videos = 0
    model_runtime = 0
    extra_runtime = 0
    global_res = ""

    # Model inference
    start_model_time = time.time()
    try:
        for _ in range(num_samples):
            generation_config = {
                "max_new_tokens": 170,
                "sampling": False,
                "stream": False,
                "max_inp_length":8192*7,
                "temperature": 0, 
            }

            prompt = "Describe this video in detail."
            msgs = [
            {
                "role": "user",
                "content": [prompt] + frames
            }
            ]

            res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **generation_config)
            global_res += res

            print(f"Generated Text for Video: {res}")

    except Exception as e:
        raise ValueError(f"Error generating for video: {e}") from e

    model_runtime += time.time() - start_model_time

    # Calculate time for additional operations
    start_extra_time = time.time()
    num_tokens = len(tokenizer.tokenize(res))
    tokens_generated += num_tokens
    num_videos += 1
    extra_runtime += time.time() - start_extra_time

    return tokens_generated, num_videos, model_runtime, extra_runtime, global_res


def benchmark_videos(config, model_id, video_paths, meta_data):
    print(f"Initializing model '{model_id}'...")
    model = load_model(model_id, config)

    tokenizer = create_tokenizer(model_id)

    if config["compile"]:
        model = torch.compile(model)

    results = []
    total_runtime = 0
    total_model_runtime = 0
    total_extra_runtime = 0
    total_queries = 0
    total_tokens = 0
    global_peak_memory_allocated = 0
    global_peak_memory_reserved = 0
    generations = ""
    
    meta_iter = metadata_generator(meta_data)

    for video_path in tqdm(video_paths, desc="Benchmarking models"):
        print(f"\nProcessing: {video_path}")
        start_time = time.time()
        
        meta = next(meta_iter)

        initial_memory_allocated = torch.cuda.memory_allocated() / 1e9
        initial_memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{os.path.basename(video_path)}] Initial Memory - Allocated: {initial_memory_allocated:.3f} GB, Reserved: {initial_memory_reserved:.3f} GB")

        tokens_generated, num_videos, model_runtime, extra_runtime, global_res = process_video(
            video_path, config["output_token_limit"], config["num_samples"], model, tokenizer, meta
        )
        peak_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
        peak_memory_reserved = torch.cuda.max_memory_reserved() / 1e9
        global_peak_memory_allocated = max(global_peak_memory_allocated, peak_memory_allocated)
        global_peak_memory_reserved = max(global_peak_memory_reserved, peak_memory_reserved)
        generations += global_res + ";"

        video_runtime = time.time() - start_time

        print(f"[{os.path.basename(video_path)}] Peak Memory - Allocated: {peak_memory_allocated:.3f} GB, Reserved: {peak_memory_reserved:.3f} GB")

        print(f"Finished {os.path.basename(video_path)}")
        print(f"  Total Runtime: {video_runtime:.2f}s")
        print(f"  Model Runtime: {model_runtime:.2f}s")
        print(f"  Extra Operations Runtime: {extra_runtime:.2f}s")
        print(f"  Tokens Generated: {tokens_generated}")
        print(f"  Queries Made/Processed Videos: {num_videos}")

        total_runtime += video_runtime
        total_model_runtime += model_runtime
        total_extra_runtime += extra_runtime
        total_queries += num_videos
        total_tokens += tokens_generated

    vps = total_queries / total_model_runtime if total_model_runtime > 0 else 0
    tps = total_tokens / total_model_runtime if total_model_runtime > 0 else 0
    tpq = total_tokens / total_queries if total_queries > 0 else 0
    video_saved = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

    results.append({
            "Timestamp": video_saved,
            "Model ID": model_id,
            "Total_Runtime": total_runtime,
            "Model_Runtime": total_model_runtime,
            "Extra_Runtime": total_extra_runtime,
            "VPS": vps,
            "TPS": tps,
            "TPQ": tpq,
            "Peak_Memory_Allocated": global_peak_memory_allocated,
            "Peak_Memory_Reserved": global_peak_memory_reserved,
            "Generations": generations,
    })

    csv_file = "benchmark_log.csv"
    csv_header = [
        "Timestamp",
        "Model ID",
        "Total_Runtime",
        "Model_Runtime",
        "Extra_Runtime",
        "VPS",
        "TPS",
        "TPQ",
        "Peak_Memory_Allocated",
        "Peak_Memory_Reserved",
        "Generations",
    ]
    
    file_exists = os.path.exists(csv_file)

    with open(csv_file, mode="a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_header)
        if not file_exists:
            writer.writeheader()
        for result in results:
            writer.writerow(result)

    print("\nBenchmark Summary:")
    print(f"  Total Runtime: {total_runtime:.2f}s")
    print(f"  Videos per Second (VPS): {vps:.2f}")
    print(f"  Tokens per Second (TPS): {tps:.2f}")
    print(f"  Tokens per Query (TPQ): {tpq:.2f}")
    print(f"  Global Peak Memory Allocated: {global_peak_memory_allocated:.3f} GB")
    print(f"  Global Peak Memory Reserved: {global_peak_memory_reserved:.3f} GB")

    torch.cuda.reset_peak_memory_stats()
    
    return

if __name__ == "__main__":

    config = parse_config("./config.yaml")
   
    #print("Extracting videos from Parquet files...")
    #video_paths = sample_n_videos(5, seed=42)
    print("ToxicAInment data used ...")
    videos, slides = get_posts()
    video_paths = []
    meta_data = []

    for v in videos:
        video_info = videos[v]
        video_paths.append(video_info["video_path"])

        meta_data.append({
        "author_name": video_info["author_name"],
        "video_description": video_info["video_description"]
       })


    for model_id in config["models"]:
        print("Benchmarking videos...")
        benchmark_videos(
            config,
            model_id,
            video_paths,
            meta_data,
        )

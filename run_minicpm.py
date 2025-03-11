import os
from pathlib import Path
import time
from datetime import datetime
import random
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from pyaml_env import parse_config
import duckdb
from download import DATASET_PATH, MP4_DATASET_PATH
from utils import get_posts, toxicainment_data_folder, video_to_frames, parse_output

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

def fill_prompt(meta_data, prompt):
    slide_desc = meta_data["author_name"]
    slide_author = meta_data["video_description"]
    filled = prompt % (slide_desc, slide_author)
    return filled

def process_video(video_path, token_limit, model, tokenizer, meta_data, slide_meta_data):

    ## TODO:
    ## pass metadata for slides
    ## a function to batch slide_paths after VideoReader processing + mp3 paths

    folder = toxicainment_data_folder()
    with open(folder / "prompt.txt", "r") as f:
        data = f.read()

    prompt_text = "".join(data)
    filled_prompt = fill_prompt(meta_data, prompt_text)

    try:
        frames, _ = video_to_frames(video_path)
    except Exception as e:
        raise ValueError(f"Error processing video: {e}") from e

    # Model inference
    generation_config = {
        "max_new_tokens": token_limit,
        "sampling": False,
        "stream": False,
        "max_inp_length":8192*7,
        "temperature": 0, 
    }

    msgs = [{"role": "user", "content": [filled_prompt] + frames}]

    start_model_time = time.time()
    try:
        res = model.chat(image=None, msgs=msgs, tokenizer=tokenizer, **generation_config)
        print(f"Generated Text for Video: {res}")

    except Exception as e:
        raise ValueError(f"Error generating for video: {e}") from e

    model_runtime = time.time() - start_model_time
    tokens_generated = len(tokenizer.tokenize(res))
    n_frames_used = len(frames)
    return tokens_generated, model_runtime, res, n_frames_used


def benchmark_videos(config, model_id, video_paths, meta_data, slide_meta_data):
    print(f"Initializing model '{model_id}'...")
    model = load_model(model_id, config)

    tokenizer = create_tokenizer(model_id)

    if config["compile"]:
        model = torch.compile(model)

    results = []
    total_runtime = 0
    total_model_runtime = 0
    total_queries = 0
    total_tokens = 0
    global_frame_count = 0
    global_peak_memory_allocated = 0
    global_peak_memory_reserved = 0
    generations = ""
    current_video = ""
    
    meta_iter = metadata_generator(meta_data)

    for video_path in tqdm(video_paths, desc="Benchmarking models"):
        print(f"\nProcessing: {video_path}")
        start_time = time.time()
        
        meta = next(meta_iter)
        current_video += video_path + ";"
        initial_memory_allocated = torch.cuda.memory_allocated() / 1e9
        initial_memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(f"[{os.path.basename(video_path)}] Initial Memory - Allocated: {initial_memory_allocated:.3f} GB, Reserved: {initial_memory_reserved:.3f} GB")

        tokens_generated, model_runtime, global_res, total_frames = process_video(
            video_path, config["output_token_limit"], model, tokenizer, meta, slide_meta_data
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
        print(f"  Tokens Generated: {tokens_generated}")

        total_runtime += video_runtime
        total_model_runtime += model_runtime
        total_queries += 1
        total_tokens += tokens_generated
        global_frame_count += total_frames

        video_saved = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        row = {
                "Timestamp": video_saved,
                "Model ID": model_id,
                "Total_Runtime": total_runtime,
                "Model_Runtime": total_model_runtime,
                "Total_Frames": total_frames,
                "Peak_Memory_Allocated": global_peak_memory_allocated,
                "Peak_Memory_Reserved": global_peak_memory_reserved,
                "Processed_Video": current_video,
                "Generations": generations,
        }

        results_file = "toxicainment_videos_log.jsonl"
        row = pd.DataFrame([row])
        
        row.to_json(results_file, orient="records", lines=True, mode="a")
        print(f"added line to {results_file}")
   
    vps = total_queries / total_model_runtime if total_model_runtime > 0 else 0
    tps = total_tokens / total_model_runtime if total_model_runtime > 0 else 0
    tpq = total_tokens / total_queries if total_queries > 0 else 0


    model_labels_csv = f"model_labels_{model_id.replace('/', '_')}.csv"
    # parse_output(csv_file, model_labels_csv, model_id)

    print("\nBenchmark Summary:")
    print(f"  Total Runtime: {total_runtime:.2f}s")
    print(f"  Videos per Second (VPS): {vps:.2f}")
    print(f"  Tokens per Second (TPS): {tps:.2f}")
    print(f"  Tokens per Query (TPQ): {tpq:.2f}")
    print(f"  Global Peak Memory Allocated: {global_peak_memory_allocated:.3f} GB")
    print(f"  Global Peak Memory Reserved: {global_peak_memory_reserved:.3f} GB")
    print(f"  Total Number of Frames(All Videos): {global_frame_count}")

    torch.cuda.reset_peak_memory_stats()
    
    return

if __name__ == "__main__":

    config = parse_config("./config.yaml")
   
    #print("Extracting videos from Parquet files...")
    #video_paths = sample_n_videos(5, seed=42)
    print("ToxicAInment data used ...")
    videos, slides = get_posts()
    video_paths = []
    slide_paths = []
    meta_data = []
    slide_meta_data = []

    for v in videos:
        video_info = videos[v]
        video_paths.append(video_info["video_path"])

        meta_data.append({
        "author_name": video_info["author_name"],
        "video_description": video_info["video_description"]
       })

    for s in slides:
        slide_info = slides[s]
        slide_paths.append(slide_info["slide_path"])

        slide_meta_data.append({
        "author_name": slide_info["author_name"],
        "slide_description": slide_info["slide_description"]
       })


    for model_id in config["models"]:
        print("Benchmarking models...")
        benchmark_videos(
            config,
            model_id,
            video_paths[:2],
            meta_data,
            slide_meta_data,
        )
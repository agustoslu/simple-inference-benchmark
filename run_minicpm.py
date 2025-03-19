from dataclasses import dataclass
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
from utils import get_posts
import argparse
from pyinstrument import Profiler
import uuid
import minicpm_omni

# install as in requirements.txt
from llmlib.huggingface_inference import video_to_imgs


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
                 WHERE id IN ({", ".join([str(id) for id in random_ids])})
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


@dataclass
class ModelAndTokenizer:
    model_id: str
    model: AutoModel
    tokenizer: AutoTokenizer
    config: dict

    def process_video(self, video_path: str, meta_data: dict):
        return process_video_minicpm(
            video_path=video_path, config=self.config, model=self, meta_data=meta_data
        )


def load_model(model_id: str, config: dict) -> ModelAndTokenizer:
    """Loads model from huggingface"""
    print(f"Initializing model '{model_id}'...")

    if model_id == minicpm_omni.model_id:
        return minicpm_omni.load_model()

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        token=config["hf_token"],
        device_map="cuda",
    ).eval()

    if config["compile"]:
        model = torch.compile(model)

    tokenizer = create_tokenizer(model_id)
    return ModelAndTokenizer(model_id, model, tokenizer, config=config)


def create_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def metadata_generator(meta_data):
    for i in meta_data:
        yield i


def fill_prompt(meta_data: dict, prompt: str) -> str:
    slide_desc = meta_data["author_name"]
    slide_author = meta_data["video_description"]
    filled = prompt % (slide_desc, slide_author)
    return filled


def process_video_minicpm(
    video_path: str, config: dict, model: ModelAndTokenizer, meta_data: dict
):
    model, tokenizer = model.model, model.tokenizer

    ## TODO:
    ## pass metadata for slides
    ## a function to batch slide_paths after VideoReader processing + mp3 paths

    with open("./prompt.txt", "r") as f:
        data = f.read()

    prompt_text = "".join(data)
    filled_prompt = fill_prompt(meta_data, prompt_text)

    try:
        frames = video_to_imgs(
            video_path, max_n_frames=config["max_n_frames_per_video"]
        )
    except Exception as e:
        raise ValueError(f"Error processing video: {e}") from e

    # Model inference
    generation_config = {
        "max_new_tokens": config["output_token_limit"],
        "sampling": True,
        "stream": False,
        "max_inp_length": 8192 * 7,
        # "temperature": 0,   # use defaults for MiniCPM
    }

    msgs = [{"role": "user", "content": [filled_prompt] + frames}]

    start_model_time = time.time()
    try:
        res = model.chat(
            image=None, msgs=msgs, tokenizer=tokenizer, **generation_config
        )
        print(f"Generated Text for Video: {res}")

    except Exception as e:
        raise ValueError(f"Error generating for video: {e}") from e

    model_runtime = time.time() - start_model_time
    n_frames_used = len(frames)
    return model_runtime, res, n_frames_used


def benchmark_videos(config, model_id, video_paths, meta_data, slide_meta_data):
    this_run = str(uuid.uuid4())
    print(f"Run ID: {this_run}")

    model = load_model(model_id, config)

    meta_iter = metadata_generator(meta_data)

    for video_path in tqdm(video_paths, desc="Benchmarking models"):
        print(f"\nProcessing: {video_path}")
        start_time = time.time()

        meta = next(meta_iter)
        initial_memory_allocated = torch.cuda.memory_allocated() / 1e9
        initial_memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(
            f"[{os.path.basename(video_path)}] Initial Memory - Allocated: {initial_memory_allocated:.3f} GB, Reserved: {initial_memory_reserved:.3f} GB"
        )

        model_runtime, response, total_frames = model.process_video(
            video_path=video_path, meta_data=meta
        )
        tokens_generated = len(model.tokenizer.tokenize(response))
        peak_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
        peak_memory_reserved = torch.cuda.max_memory_reserved() / 1e9

        video_runtime = time.time() - start_time

        print(
            f"[{os.path.basename(video_path)}] Peak Memory - Allocated: {peak_memory_allocated:.3f} GB, Reserved: {peak_memory_reserved:.3f} GB"
        )

        print(f"Finished {os.path.basename(video_path)}")
        print(f"  Total Runtime: {video_runtime:.2f}s")
        print(f"  Model Runtime: {model_runtime:.2f}s")
        print(f"  Tokens Generated: {tokens_generated}")

        video_saved = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "Run_ID": this_run,
            "Timestamp": video_saved,
            "Model ID": model_id,
            "Total_Runtime": video_runtime,
            "Model_Runtime": model_runtime,
            "Tokens_Generated": tokens_generated,
            "Total_Frames": total_frames,
            "Peak_Memory_Allocated": peak_memory_allocated,
            "Peak_Memory_Reserved": peak_memory_reserved,
            "Processed_Video": video_path,
            "Generations": response,
        }

        results_file = "toxicainment_videos_log.jsonl"
        row = pd.DataFrame([row])

        row.to_json(results_file, orient="records", lines=True, mode="a")
        print(f"added line to {results_file}")

    torch.cuda.reset_peak_memory_stats()
    return


def run_benchmark(n_examples: int = -1, models=None) -> None:
    config = parse_config("./config.yaml")

    print("ToxicAInment data used ...")
    videos, slides = get_posts(n_examples)
    video_paths = []
    slide_paths = []
    meta_data = []
    slide_meta_data = []

    for v in videos:
        video_info = videos[v]
        video_paths.append(video_info["video_path"])

        meta_data.append(
            {
                "author_name": video_info["author_name"],
                "video_description": video_info["video_description"],
            }
        )

    for s in slides:
        slide_info = slides[s]
        slide_paths.append(slide_info["slide_path"])

        slide_meta_data.append(
            {
                "author_name": slide_info["author_name"],
                "slide_description": slide_info["slide_description"],
            }
        )

    if models is None:
        models = config["models"]

    for model_id in models:
        benchmark_videos(
            config,
            model_id,
            video_paths,
            meta_data,
            slide_meta_data,
        )


def parse_command_line_args():
    parser = argparse.ArgumentParser(description="Run MiniCPM benchmark")
    parser.add_argument(
        "--profile", action="store_true", default=False, help="Enable profiling"
    )
    args = parser.parse_args()
    print(f"Profile: {args.profile}")
    return args


if __name__ == "__main__":
    args = parse_command_line_args()
    if args.profile:
        profiler = Profiler(interval=0.01)
        with profiler:
            run_benchmark(n_examples=5, models=["openbmb/MiniCPM-V-2_6"])
        os.makedirs("profiles", exist_ok=True)
        profiler.write_html("profiles/minicpm_profile.html")
    else:
        run_benchmark(n_examples=2)  # 2 vids just to check everything runs

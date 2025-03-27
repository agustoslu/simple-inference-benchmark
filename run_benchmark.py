from dataclasses import dataclass
import os
from pathlib import Path
import time
from datetime import datetime
import random
from typing import Literal
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import duckdb
from download import DATASET_PATH, MP4_DATASET_PATH, read_prompt_template, code_root
from utils import get_posts
from pydantic_settings import BaseSettings
from pyinstrument import Profiler
import uuid
import logging


# install llmlib as described in the README.md
from llmlib.huggingface_inference import video_to_imgs
from llmlib.gemma3_local import Gemma3Local, Message
from llmlib.gemma3_vllm import Gemma3vLLM as Gemma3vLLM_llmlib, Conversation
from llmlib.qwen2_5 import Qwen2_5


class BenchmarkArgs(BaseSettings, cli_parse_args=True):
    profile: bool = False
    model_id: str = "openbmb/MiniCPM-V-2_6"
    n_examples: int = 2
    use_vllm: bool = False
    max_n_frames_per_video: int = 50
    output_token_limit: int = 512
    compile: bool = False
    gpu_size: Literal["24GB", "80GB"] = "24GB"


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
class VideoOutput:
    response: str
    n_frames_used: int
    model_runtime: float


@dataclass
class ModelInterface:
    model_id: str
    args: BenchmarkArgs

    def process_video(self, video_path: str, meta_data: dict) -> VideoOutput:
        raise NotImplementedError("Subclasses can implement this method")

    def process_batch_of_videos(
        self, video_paths: list[Path], meta_data: list[dict]
    ) -> list[VideoOutput]:
        raise NotImplementedError("Subclasses can implement this method")


@dataclass
class MiniCPM(ModelInterface):
    model: AutoModel
    tokenizer: AutoTokenizer

    def process_video(self, video_path: str, meta_data: dict) -> VideoOutput:
        return process_video_minicpm(
            video_path=video_path, args=self.args, model=self, meta_data=meta_data
        )


@dataclass
class Gemma3Hf(ModelInterface):
    llmlib_model: Gemma3Local

    def process_video(self, video_path: str, meta_data: dict) -> VideoOutput:
        prompt_template = read_prompt_template()
        filled_prompt = fill_prompt(meta_data=meta_data, prompt=prompt_template)
        messages = [Message(role="user", msg=filled_prompt, video=video_path)]
        output = self.llmlib_model.complete_msgs(msgs=messages, output_dict=True)
        return VideoOutput(
            response=output["response"],
            n_frames_used=output["n_frames"],
            model_runtime=output["model_runtime"],
        )


@dataclass
class Gemma3vLLM(ModelInterface):
    llmlib_model: Gemma3vLLM_llmlib

    def process_batch_of_videos(
        self, video_paths: list[Path], meta_data: list[dict]
    ) -> list[VideoOutput]:
        assert len(video_paths) == len(meta_data)

        convos: list[Conversation] = []
        for video_path, meta in zip(video_paths, meta_data):
            prompt_template = read_prompt_template()
            filled_prompt = fill_prompt(meta_data=meta, prompt=prompt_template)
            conversation = [Message(role="user", msg=filled_prompt, video=video_path)]
            convos.append(conversation)

        output = self.llmlib_model.complete_batch(batch=convos, output_dict=True)
        return [
            VideoOutput(
                response=output["response"],
                n_frames_used=output["n_frames"],
                model_runtime=output["model_runtime"],
            )
            for output in output
        ]


@dataclass
class Qwen(ModelInterface):
    llmlib_model: Qwen2_5

    def process_video(self, video_path: str, meta_data: dict) -> VideoOutput:
        prompt_template = read_prompt_template()
        filled_prompt = fill_prompt(meta_data=meta_data, prompt=prompt_template)
        messages = [Message(role="user", msg=filled_prompt, video=video_path)]
        output = self.llmlib_model.complete_msgs(msgs=messages, output_dict=True)
        return VideoOutput(
            response=output["response"],
            n_frames_used=output["n_frames"],
            model_runtime=output["model_runtime"],
        )


def enable_info_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model(args: BenchmarkArgs) -> ModelInterface:
    """Loads model from huggingface"""
    model_id = args.model_id
    print(f"Initializing model '{model_id}'...")

    # if model_id == minicpm_omni.model_id:
    #     return minicpm_omni.load_model()

    if "gemma-3" in model_id:
        if args.use_vllm:
            return load_gemma3_vllm(args)
        else:  # use huggingface
            return load_gemma3_huggingface(args)

    if "Qwen" in model_id:
        return load_qwen(args)

    model = AutoModel.from_pretrained(
        model_id,
        trust_remote_code=True,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        token=os.environ["HF_TOKEN"],
        device_map="cuda",
    ).eval()

    if args.compile:
        model = torch.compile(model)

    tokenizer = create_tokenizer(model_id)
    return MiniCPM(model_id=model_id, model=model, tokenizer=tokenizer, args=args)


def load_gemma3_huggingface(args: BenchmarkArgs) -> Gemma3Hf:
    llmlib_model = Gemma3Local(
        model_id=args.model_id,
        max_n_frames_per_video=args.max_n_frames_per_video,
        max_new_tokens=args.output_token_limit,
    )
    return Gemma3Hf(model_id=args.model_id, args=args, llmlib_model=llmlib_model)


def load_gemma3_vllm(args: BenchmarkArgs) -> Gemma3vLLM:
    llmlib_model = Gemma3vLLM_llmlib(
        model_id=args.model_id,
        max_n_frames_per_video=args.max_n_frames_per_video,
        max_new_tokens=args.output_token_limit,
        gpu_size=args.gpu_size,
    )
    return Gemma3vLLM(model_id=args.model_id, args=args, llmlib_model=llmlib_model)


def load_qwen(args: BenchmarkArgs) -> Qwen:
    llmlib_model = Qwen2_5(
        model_id=args.model_id,
        max_n_frames_per_video=args.max_n_frames_per_video,
        max_new_tokens=args.output_token_limit,
    )
    qwen = Qwen(
        model_id=args.model_id,
        model=llmlib_model.model,
        tokenizer=llmlib_model.processor.tokenizer,
        args=args,
        llmlib_model=llmlib_model,
    )
    return qwen


def create_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


def fill_prompt(meta_data: dict, prompt: str) -> str:
    slide_desc = meta_data["author_name"]
    slide_author = meta_data["video_description"]
    filled = prompt % (slide_desc, slide_author)
    return filled


def process_video_minicpm(
    video_path: str, args: BenchmarkArgs, model: ModelInterface, meta_data: dict
) -> VideoOutput:
    model, tokenizer = model.model, model.tokenizer

    ## TODO:
    ## pass metadata for slides
    ## a function to batch slide_paths after VideoReader processing + mp3 paths

    prompt_template = read_prompt_template()

    prompt_text = "".join(prompt_template)
    filled_prompt = fill_prompt(meta_data, prompt_text)

    try:
        frames = video_to_imgs(video_path, max_n_frames=args.max_n_frames_per_video)
    except Exception as e:
        raise ValueError(f"Error processing video: {e}") from e

    # Model inference
    generation_config = {
        "max_new_tokens": args.output_token_limit,
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
    return VideoOutput(
        response=res, n_frames_used=n_frames_used, model_runtime=model_runtime
    )


def benchmark_videos(
    args: BenchmarkArgs,
    video_paths: list[Path],
    meta_data: list[dict],
    slide_meta_data: list[dict],
):
    model = load_model(args)
    if args.use_vllm:
        batch_process_dataset(model=model, video_paths=video_paths, meta_data=meta_data)
    else:
        process_dataset_by_row(
            model=model, video_paths=video_paths, meta_data=meta_data
        )


def process_dataset_by_row(
    model: ModelInterface, video_paths: list[Path], meta_data: list[dict]
):
    this_run = str(uuid.uuid4())
    print(f"Run ID: {this_run}")

    for video_path, meta in tqdm(
        zip(video_paths, meta_data), desc="Benchmarking model"
    ):
        print(f"\nProcessing: {video_path.name}")
        start_time = time.time()

        initial_memory_allocated = torch.cuda.memory_allocated() / 1e9
        initial_memory_reserved = torch.cuda.memory_reserved() / 1e9
        print(
            f"[{video_path.name}] Initial Memory - Allocated: {initial_memory_allocated:.3f} GB, Reserved: {initial_memory_reserved:.3f} GB"
        )

        output = model.process_video(video_path=video_path, meta_data=meta)
        tokens_generated = len(model.tokenizer.tokenize(output.response))
        peak_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
        peak_memory_reserved = torch.cuda.max_memory_reserved() / 1e9

        video_runtime = time.time() - start_time

        print(
            f"[{video_path.name}] Peak Memory - Allocated: {peak_memory_allocated:.3f} GB, Reserved: {peak_memory_reserved:.3f} GB"
        )

        print(f"Finished {video_path.name}")
        print(f"  Total Runtime: {video_runtime:.2f}s")
        print(f"  Model Runtime: {output.model_runtime:.2f}s")
        print(f"  Num Frames:  {output.n_frames_used}")
        print(f"  Tokens Generated: {tokens_generated}")

        video_saved = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        row = {
            "Run_ID": this_run,
            "Timestamp": video_saved,
            "Model ID": model.model_id,
            "Total_Runtime": video_runtime,
            "Model_Runtime": output.model_runtime,
            "Tokens_Generated": tokens_generated,
            "Total_Frames": output.n_frames_used,
            "Peak_Memory_Allocated": peak_memory_allocated,
            "Peak_Memory_Reserved": peak_memory_reserved,
            "Processed_Video": video_path.name,
            "Generations": output.response,
        }

        results_file = code_root / "toxicainment_videos_log.jsonl"
        row = pd.DataFrame([row])

        row.to_json(results_file, orient="records", lines=True, mode="a")
        print(f"added line to {results_file}")
        torch.cuda.reset_peak_memory_stats()


def batch_process_dataset(
    model: ModelInterface, video_paths: list[Path], meta_data: list[dict]
):
    assert isinstance(model, Gemma3vLLM), type(model)
    results = model.process_batch_of_videos(
        video_paths=video_paths, meta_data=meta_data
    )
    # TODO: save results to file


def run_benchmark(args: BenchmarkArgs) -> None:
    print("ToxicAInment data used ...")
    videos, slides = get_posts(args.n_examples)
    video_paths: list[Path] = []
    slide_paths = []
    meta_data = []
    slide_meta_data = []

    for v in videos:
        video_info = videos[v]
        video_paths.append(Path(video_info["video_path"]))

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

    benchmark_videos(args, video_paths, meta_data, slide_meta_data)


if __name__ == "__main__":
    enable_info_logs()
    args = BenchmarkArgs()
    print(args)
    if args.profile:
        profiler = Profiler(interval=0.01)
        with profiler:
            run_benchmark(args)
        os.makedirs("profiles", exist_ok=True)
        profiler.write_html(
            f"profiles/profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )
    else:
        run_benchmark(args)

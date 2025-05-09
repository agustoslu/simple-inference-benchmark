from dataclasses import dataclass
import os
from pathlib import Path
import time
from datetime import datetime
import random
from typing import Literal, Iterable
from bench_lib.utils import (
    enable_info_logs,
    fill_prompt,
    read_prompt_template,
    to_dataset,
)
import pandas as pd
from pydantic import BaseModel
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
import duckdb
from download import DATASET_PATH, MP4_DATASET_PATH, code_root
from bench_lib.utils import get_posts_df
from pydantic_settings import BaseSettings
from pyinstrument import Profiler
import uuid
import logging

# install llmlib as described in the README.md
from llmlib.huggingface_inference import video_to_imgs
from llmlib.gemma3_local import Gemma3Local, Message
from llmlib.vllm_model import ModelvLLM
from llmlib.qwen2_5 import Qwen2_5
from llmlib.llama_4 import Llama_4
from llmlib.gemini.gemini_code import GeminiAPI


logger = logging.getLogger(__name__)


class BenchmarkArgs(BaseSettings, cli_parse_args=True):
    profile: bool = False
    model_id: str = "google/gemma-3-4b-it"
    n_examples: int = 2
    use_vllm: bool = False
    max_n_frames_per_video: int = 50
    output_token_limit: int = 512
    compile: bool = False
    gpu_size: Literal["24GB", "80GB"] = "24GB"
    restart: bool = False

    vllm_max_model_len: int = 8192


# Extract and sample videos
def sample_n_videos(n: int, seed: int):
    con = duckdb.connect()
    n_videos = con.sql(f"SELECT COUNT(*) FROM '{DATASET_PATH}/*.parquet'").fetchone()[0]  # type: ignore
    logger.info("Total videos: %d", n_videos)
    random.seed(seed)
    random_ids = random.sample(range(n_videos), n)
    logger.info("Randomly selected video IDs: %s", random_ids)
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
            logger.info("Saved video to %s", mp4_video_path)
        video_paths.append(mp4_video_path)

    return video_paths


# TODO: Move to llmlib
@dataclass
class VideoOutput:
    response: str
    n_frames_used: int | None
    model_runtime: float | None
    post_id: str | None = None
    video_path: str | None = None


@dataclass
class ModelInterface:
    model_id: str
    args: BenchmarkArgs

    def process_video(self, video_path: str | Path, meta_data: dict) -> VideoOutput:
        raise NotImplementedError("Subclasses can implement this method")

    def process_batch_of_videos(
        self, video_paths: list[Path], meta_data: list[dict]
    ) -> Iterable[VideoOutput]:
        raise NotImplementedError("Subclasses can implement this method")


@dataclass
class HuggingFaceModel(ModelInterface):
    model: AutoModel
    tokenizer: AutoTokenizer


@dataclass
class MiniCPM(HuggingFaceModel):
    def process_video(self, video_path: str | Path, meta_data: dict) -> VideoOutput:
        return process_video_minicpm(
            video_path=video_path, args=self.args, hf=self, meta_data=meta_data
        )


@dataclass
class Gemma3Hf(HuggingFaceModel):
    llmlib_model: Gemma3Local

    def process_video(self, video_path: str | Path, meta_data: dict) -> VideoOutput:
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
class ModelvLLM_Benchmark(ModelInterface):
    llmlib_model: ModelvLLM

    def process_batch_of_videos(self, posts_df: pd.DataFrame) -> Iterable[VideoOutput]:
        all_req_ids = [str(i) for i in range(len(posts_df))]
        posts_df = posts_df.assign(request_id=all_req_ids)
        posts_df.set_index("request_id", inplace=True)

        dataset = to_dataset(posts_df)
        gen = self.llmlib_model.complete_batch(batch=dataset, output_dict=True)
        for output_dict in gen:
            req_id = output_dict["request_id"]
            vo = VideoOutput(
                response=output_dict["response"],
                n_frames_used=output_dict["n_frames"],
                model_runtime=output_dict["model_runtime"],
                post_id=posts_df.loc[req_id, "video_id"],
                video_path=str(posts_df.loc[req_id, "video_path"]),
            )
            yield vo


@dataclass
class Qwen(HuggingFaceModel):
    llmlib_model: Qwen2_5

    def process_video(self, video_path: str | Path, meta_data: dict) -> VideoOutput:
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
class Llama(HuggingFaceModel):
    llmlib_model: Llama_4

    def process_video(self, video_path: str | Path, meta_data: dict) -> VideoOutput:
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
class Gemini(ModelInterface):
    llmlib_model: GeminiAPI

    def process_video(self, video_path: str | Path, meta_data: dict) -> VideoOutput:
        prompt_template = read_prompt_template()
        filled_prompt = fill_prompt(meta_data=meta_data, prompt=prompt_template)
        response = self.llmlib_model.video_prompt(
            video=video_path, prompt=filled_prompt
        )
        return VideoOutput(
            response=response,
            n_frames_used=None,  # Gemini handles video frames internally
            model_runtime=None,  # Remote API doesn't provide model runtime
        )


def load_model(args: BenchmarkArgs) -> ModelInterface:
    """Loads model from huggingface"""
    model_id = args.model_id
    logger.info("Initializing model '%s'...", model_id)

    # if model_id == minicpm_omni.model_id:
    #     return minicpm_omni.load_model()

    if args.use_vllm:
        return load_vllm_model(args)

    if "gemma-3" in model_id:
        return load_gemma3_huggingface(args)

    if "Qwen" in model_id:
        return load_qwen(args)

    if "Llama" in model_id:
        return load_llama(args)

    if "gemini" in model_id.lower():
        return load_gemini(args)

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
    return Gemma3Hf(
        model=llmlib_model.model,
        tokenizer=llmlib_model.processor.tokenizer,
        model_id=args.model_id,
        args=args,
        llmlib_model=llmlib_model,
    )


def load_vllm_model(args: BenchmarkArgs) -> ModelvLLM_Benchmark:
    llmlib_model = ModelvLLM(
        model_id=args.model_id,
        max_n_frames_per_video=args.max_n_frames_per_video,
        max_new_tokens=args.output_token_limit,
        gpu_size=args.gpu_size,
        max_model_len=args.vllm_max_model_len,
    )
    return ModelvLLM_Benchmark(
        model_id=args.model_id, args=args, llmlib_model=llmlib_model
    )


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


def load_llama(args: BenchmarkArgs) -> Llama:
    llmlib_model = Llama_4(
        model_id=args.model_id,
        max_n_frames_per_video=args.max_n_frames_per_video,
        max_new_tokens=args.output_token_limit,
    )
    llama = Llama(
        model_id=args.model_id,
        model=llmlib_model.model,
        tokenizer=llmlib_model.processor.tokenizer,
        args=args,
        llmlib_model=llmlib_model,
    )
    return llama


def load_gemini(args: BenchmarkArgs) -> Gemini:
    llmlib_model = GeminiAPI(
        model_id=args.model_id,
        max_output_tokens=args.output_token_limit,
        location="us-central1",
        delete_files_after_use=False,
        json_schema=None,  # SaxonyDeletedContentSchema,
    )
    return Gemini(model_id=args.model_id, args=args, llmlib_model=llmlib_model)


def create_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


class Answer(BaseModel):
    question: str
    comment: str
    answer: Literal["yes", "no"]


class SaxonyDeletedContentSchema(BaseModel):
    answers: list[Answer]


def process_video_minicpm(
    video_path: str | Path, args: BenchmarkArgs, hf: HuggingFaceModel, meta_data: dict
) -> VideoOutput:
    model, tokenizer = hf.model, hf.tokenizer

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
        logger.info("Generated Text for Video: %s", res)

    except Exception as e:
        raise ValueError(f"Error generating for video: {e}") from e

    model_runtime = time.time() - start_model_time
    n_frames_used = len(frames)
    return VideoOutput(
        response=res, n_frames_used=n_frames_used, model_runtime=model_runtime
    )


def benchmark_videos(args: BenchmarkArgs, posts_df: pd.DataFrame):
    model = load_model(args)
    if args.use_vllm:
        batch_process_dataset(model=model, posts_df=posts_df)
    elif isinstance(model, HuggingFaceModel):
        process_dataset_by_row(model=model, posts_df=posts_df)
    else:  # Remote models like Gemini
        process_dataset_by_row_remotely(model=model, posts_df=posts_df)


def process_dataset_by_row_remotely(model: ModelInterface, posts_df: pd.DataFrame):
    this_run = generate_run_uuid()

    for _, row in tqdm(list(posts_df.iterrows()), desc="Benchmarking model"):
        video_path = row["video_path"]
        logger.info("\nProcessing: %s", video_path)

        start_time = time.time()
        output = model.process_video(video_path=video_path, meta_data=row.to_dict())
        video_runtime = time.time() - start_time

        logger.info("Finished %s", video_path.name)
        logger.info("  Total Runtime: %.2fs", video_runtime)
        logger.info("  Response: %s...", output.response[:100])

        row = {
            "Run_ID": this_run,
            "Timestamp": generate_timestamp(),
            "Model ID": model.model_id,
            "Total_Runtime": video_runtime,
            "Processed_Video": video_path.name,
            "Generations": output.response,
            "video_id": row["video_id"],
        }

        df = pd.DataFrame([row])
        save_to_results_files(df)


def process_dataset_by_row(model: HuggingFaceModel, posts_df: pd.DataFrame):
    this_run = generate_run_uuid()

    for _, row in tqdm(list(posts_df.iterrows()), desc="Benchmarking model"):
        video_path = row["video_path"]
        logger.info("\nProcessing: %s", video_path)
        start_time = time.time()
        initial_memory_allocated = torch.cuda.memory_allocated() / 1e9
        initial_memory_reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(
            "[%s] Initial Memory - Allocated: %.3f GB, Reserved: %.3f GB",
            video_path.name,
            initial_memory_allocated,
            initial_memory_reserved,
        )

        output = model.process_video(video_path=video_path, meta_data=row.to_dict())
        tokens_generated = len(model.tokenizer.tokenize(str((output.response))))
        peak_memory_allocated = torch.cuda.max_memory_allocated() / 1e9
        peak_memory_reserved = torch.cuda.max_memory_reserved() / 1e9

        video_runtime = time.time() - start_time

        logger.info(
            "[%s] Peak Memory - Allocated: %.3f GB, Reserved: %.3f GB",
            video_path.name,
            peak_memory_allocated,
            peak_memory_reserved,
        )

        logger.info("Finished %s", video_path.name)
        logger.info("  Total Runtime: %.2fs", video_runtime)
        logger.info("  Model Runtime: %.2fs", output.model_runtime)
        logger.info("  Num Frames:  %d", output.n_frames_used)
        logger.info("  Tokens Generated: %d", tokens_generated)

        row = {
            "Run_ID": this_run,
            "Timestamp": generate_timestamp(),
            "Model ID": model.model_id,
            "Total_Runtime": video_runtime,
            "Model_Runtime": output.model_runtime,
            "Tokens_Generated": tokens_generated,
            "Total_Frames": output.n_frames_used,
            "Peak_Memory_Allocated": peak_memory_allocated,
            "Peak_Memory_Reserved": peak_memory_reserved,
            "Processed_Video": video_path.name,
            "Generations": output.response,
            "video_id": row["video_id"],
        }

        df = pd.DataFrame([row])
        save_to_results_files(df)
        torch.cuda.reset_peak_memory_stats()


def generate_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_run_uuid() -> str:
    uid = str(uuid.uuid4())
    logger.info("Run ID: %s", uid)
    return uid


def save_to_results_files(df: pd.DataFrame) -> None:
    df.to_json(results_file(), orient="records", lines=True, mode="a")
    logger.info("added line to %s", results_file())


def results_file() -> Path:
    return code_root / "toxicainment_videos_log.jsonl"


def batch_process_dataset(model: ModelInterface, posts_df: pd.DataFrame):
    assert isinstance(model, ModelvLLM_Benchmark), type(model)
    gen: Iterable[VideoOutput] = model.process_batch_of_videos(posts_df)
    run_id = generate_run_uuid()
    for video_output in gen:
        row = {
            "Run_ID": run_id,
            "Timestamp": generate_timestamp(),
            "Model ID": model.model_id,
            "Total_Runtime": video_output.model_runtime,
            "Model_Runtime": video_output.model_runtime,
            # TODO: "Tokens_Generated": [r.tokens_generated for r in results],
            "Total_Frames": video_output.n_frames_used,
            "Processed_Video": video_output.video_path,
            "Generations": video_output.response,
            "video_id": video_output.post_id,
        }
        df = pd.DataFrame([row])
        save_to_results_files(df)


def run_benchmark(args: BenchmarkArgs) -> None:
    logger.info("ToxicAInment data used ...")
    posts_df = get_posts_df()
    if args.restart:
        posts_df = discard_posts_already_processed(posts_df)
    posts_df = posts_df.head(args.n_examples)
    benchmark_videos(args, posts_df)


def discard_posts_already_processed(posts_df: pd.DataFrame) -> pd.DataFrame:
    already_processed = pd.read_json(results_file(), orient="records", lines=True)
    logger.info(
        "Already processed %d posts", already_processed["Processed_Video"].nunique()
    )
    # TODO: Replace this ID parsing with plain access to df["video_id"] (once all generations have it)
    processed_ids = already_processed["Processed_Video"].str[-23:-4]
    posts_df = posts_df[~posts_df["video_id"].isin(processed_ids)]
    logger.info("%d posts remain to be processed", len(posts_df))
    return posts_df


if __name__ == "__main__":
    enable_info_logs()
    args = BenchmarkArgs()
    logger.info("%s", args)
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

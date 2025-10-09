from dataclasses import dataclass
import json
import os
from pathlib import Path
import time
from datetime import datetime
from typing import Literal, Iterable, Generator
from bench_lib.utils import (
    enable_info_logs,
    fill_prompt,
    read_prompt_template,
    saxony_dataset_dir,
    to_iterof_llmreqs,
)
from bench_lib.io import (
    Input,
    OnlyVideo,
    TranscribedVideo,
    MutedVideo,
    MutedNoTranscriptVideo,
    Output,
    strategy_map,
)
import pandas as pd
from pydantic import BaseModel
import torch
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from bench_lib.utils import get_posts_df
from pydantic_settings import BaseSettings
from pyinstrument import Profiler
import uuid
import logging
from contextlib import contextmanager

# install llmlib as described in the README.md
from llmlib.huggingface_inference import video_to_imgs
from llmlib.gemma3_local import Gemma3Local
from llmlib.vllm_model import ModelvLLM
from llmlib.qwen2_5 import Qwen2_5
from llmlib.qwen_2_5_omni import Qwen2_5_Omni
from llmlib.llama_4 import Llama_4
from llmlib.gemini.gemini_code import GeminiAPI
from llmlib.base_llm import Message, LlmReq
from llmlib.vllmserver import spinup_vllm_server


logger = logging.getLogger(__name__)


class BenchmarkArgs(BaseSettings, cli_parse_args=True):
    profile: bool = False
    model_id: str = "google/gemma-3-4b-it"
    n_examples: int = 2
    use_vllm: bool = False
    max_n_frames_per_video: int = 50
    output_token_limit: int = 512
    compile: bool = False
    restart: bool = False
    dataset_dir: Path = saxony_dataset_dir()
    tgt_file: Path = "responses.jsonl"
    input_strategy: str = "muted"

    vllm_start_server: bool = False
    vllm_port: int = 8000
    vllm_remote_call_concurrency: int = 8
    vllm_allowed_local_media_path: str = "/home/"
    vllm_tensor_parallel_size: int = 1


@dataclass
class ModelInterface:
    model_id: str
    args: BenchmarkArgs
    input_strategy: Input

    def process_video(
        self, video_path: str | Path, meta_data: dict, **kwargs
    ) -> Output:
        raise NotImplementedError("Subclasses can implement this method")


@dataclass
class HuggingFaceModel(ModelInterface):
    model: AutoModel
    tokenizer: AutoTokenizer


@dataclass
class MiniCPM(HuggingFaceModel):
    def process_video(
        self, video_path: str | Path, meta_data: dict, **kwargs
    ) -> Output:
        return process_video_minicpm(
            video_path=video_path,
            args=self.args,
            hf=self,
            meta_data=meta_data,
            **kwargs,
        )


@dataclass
class Gemma3Hf(HuggingFaceModel):
    llmlib_model: Gemma3Local

    def process_video(
        self, video_path: str | Path, meta_data: dict, **kwargs
    ) -> Output:
        prepared = self.input_strategy.prepare(
            video_path=video_path, meta_data=meta_data, **kwargs
        )
        output = self.llmlib_model.complete_msgs(
            msgs=prepared["messages"], output_dict=True
        )
        return Output(response=output["response"])


@dataclass
class ModelvLLM_Benchmark(ModelInterface):
    llmlib_model: ModelvLLM


@dataclass
class Qwen(HuggingFaceModel):
    llmlib_model: Qwen2_5

    def process_video(
        self, video_path: str | Path, meta_data: dict, **kwargs
    ) -> Output:
        prepared = self.input_strategy.prepare(
            video_path=video_path, meta_data=meta_data, **kwargs
        )
        output = self.llmlib_model.complete_msgs(
            msgs=prepared["messages"], output_dict=True
        )
        return Output(response=output["response"])


@dataclass
class Qwen_Omni(HuggingFaceModel):
    llmlib_model: Qwen2_5_Omni

    def process_video(
        self, video_path: str | Path, meta_data: dict, **kwargs
    ) -> Output:
        prepared = self.input_strategy.prepare(
            video_path=video_path, meta_data=meta_data, **kwargs
        )
        output = self.llmlib_model.complete_msgs(
            msgs=prepared["messages"], output_dict=True
        )
        return Output(response=output["response"])


@dataclass
class Llama(HuggingFaceModel):
    llmlib_model: Llama_4

    def process_video(
        self, video_path: str | Path, meta_data: dict, **kwargs
    ) -> Output:
        prepared = self.input_strategy.prepare(
            video_path=video_path, meta_data=meta_data, **kwargs
        )
        output = self.llmlib_model.complete_msgs(
            msgs=prepared["messages"], output_dict=True
        )
        return Output(response=output["response"])


@dataclass
class Gemini(ModelInterface):
    llmlib_model: GeminiAPI

    def process_video(
        self, video_path: str | Path, meta_data: dict, **kwargs
    ) -> Output:
        prepared = self.input_strategy.prepare(
            video_path=video_path, meta_data=meta_data, **kwargs
        )
        output = self.llmlib_model.complete_msgs(
            msgs=prepared["messages"], output_dict=True
        )
        return Output(response=output["response"])


def load_model(args: BenchmarkArgs, input_strategy: Input) -> ModelInterface:
    """Loads model from huggingface"""
    model_id = args.model_id
    if args.use_vllm:
        return load_vllm_model(args, input_strategy)

    logger.info("Initializing model '%s'...", model_id)

    if "gemma-3" in model_id:
        return load_gemma3_huggingface(args, input_strategy)

    if "Qwen2.5-VL" in model_id:
        return load_qwen(args, input_strategy)

    if "Qwen2.5-Omni" in model_id:
        return load_qwen_omni(args, input_strategy)

    if "Llama" in model_id:
        return load_llama(args, input_strategy)

    if "gemini" in model_id.lower():
        return load_gemini(args, input_strategy)

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
    return MiniCPM(
        model_id=model_id,
        model=model,
        tokenizer=tokenizer,
        args=args,
        input_strategy=input_strategy,
    )


def load_gemma3_huggingface(args: BenchmarkArgs, input_strategy: Input) -> Gemma3Hf:
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
        input_strategy=input_strategy,
    )


def load_vllm_model(args: BenchmarkArgs, input_strategy: Input) -> ModelvLLM_Benchmark:
    logger.info("Using vLLM model '%s'...", args.model_id)
    llmlib_model = ModelvLLM(
        model_id=args.model_id,
        max_new_tokens=args.output_token_limit,
        remote_call_concurrency=args.vllm_remote_call_concurrency,
        port=args.vllm_port,
        timeout_secs=300,
    )
    return ModelvLLM_Benchmark(
        model_id=args.model_id,
        args=args,
        llmlib_model=llmlib_model,
        input_strategy=input_strategy,
    )


def load_qwen(args: BenchmarkArgs, input_strategy: Input) -> Qwen:
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
        input_strategy=input_strategy,
    )
    return qwen


def load_qwen_omni(args: BenchmarkArgs, input_strategy: Input) -> Qwen_Omni:
    llmlib_model = Qwen2_5_Omni(
        model_id=args.model_id,
        max_n_frames_per_video=args.max_n_frames_per_video,
        max_new_tokens=args.output_token_limit,
    )
    qwen_omni = Qwen_Omni(
        model_id=args.model_id,
        model=llmlib_model.model,
        tokenizer=llmlib_model.processor.tokenizer,
        args=args,
        llmlib_model=llmlib_model,
        input_strategy=input_strategy,
    )
    return qwen_omni


def load_llama(args: BenchmarkArgs, input_strategy: Input) -> Llama:
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
        input_strategy=input_strategy,
    )
    return llama


def load_gemini(args: BenchmarkArgs, input_strategy: Input) -> Gemini:
    llmlib_model = GeminiAPI(
        model_id=args.model_id,
        max_output_tokens=args.output_token_limit,
        location="us-central1",
        delete_files_after_use=False,
        json_schema=None,  # SaxonyDeletedContentSchema,
    )
    return Gemini(
        model_id=args.model_id,
        args=args,
        llmlib_model=llmlib_model,
        input_strategy=input_strategy,
    )


def create_tokenizer(model_id):
    return AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


class Answer(BaseModel):
    question: str
    comment: str
    answer: Literal["yes", "no"]


class SaxonyDeletedContentSchema(BaseModel):
    answers: list[Answer]


def process_video_minicpm(
    video_path: str | Path,
    args: BenchmarkArgs,
    hf: HuggingFaceModel,
    meta_data: dict,
    **kwargs,
) -> Output:
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

    try:
        res = model.chat(
            image=None, msgs=msgs, tokenizer=tokenizer, **generation_config
        )
        logger.info("Generated Text for Video: %s", res)

    except Exception as e:
        raise ValueError(f"Error generating for video: {e}") from e

    return Output(response=res)


def benchmark_videos(args: BenchmarkArgs, posts_df: pd.DataFrame):
    input_strategy = strategy_map[args.input_strategy]()
    model = load_model(args, input_strategy=input_strategy)
    if args.use_vllm:
        batch_process_dataset(model=model, posts_df=posts_df, tgt_file=args.tgt_file)
    elif isinstance(model, HuggingFaceModel):
        process_dataset_by_row(model=model, posts_df=posts_df, tgt_file=args.tgt_file)
    else:  # Remote models like Gemini
        process_dataset_by_row_remotely(
            model=model, posts_df=posts_df, tgt_file=args.tgt_file
        )


def process_dataset_by_row_remotely(
    model: ModelInterface, posts_df: pd.DataFrame, tgt_file: Path
):
    this_run = generate_run_uuid()

    for _, row in tqdm(list(posts_df.iterrows()), desc="Benchmarking model"):
        video_path = row["filenames"][0]
        logger.info("\nProcessing: %s", video_path)

        start_time = time.time()
        output = model.process_video(
            video_path=video_path,
            meta_data=row.to_dict(),
            transcript=row.get("transcript"),
            dataset_dir=args.dataset_dir,
            video_id=row["video_id"],
        )
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
        save_to_results_files(df, tgt_file)


def process_dataset_by_row(
    model: HuggingFaceModel, posts_df: pd.DataFrame, tgt_file: Path
):
    this_run = generate_run_uuid()

    for _, row in tqdm(list(posts_df.iterrows()), desc="Benchmarking model"):
        video_path = row["filenames"][0]
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

        output = model.process_video(
            video_path=video_path,
            meta_data=row.to_dict(),
            transcript=row.get("transcript"),
            dataset_dir=args.dataset_dir,
            video_id=row["video_id"],
        )
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
        logger.info("  Tokens Generated: %d", tokens_generated)

        row = {
            "Run_ID": this_run,
            "Timestamp": generate_timestamp(),
            "Model ID": model.model_id,
            "Total_Runtime": video_runtime,
            "Tokens_Generated": tokens_generated,
            "Peak_Memory_Allocated": peak_memory_allocated,
            "Peak_Memory_Reserved": peak_memory_reserved,
            "Processed_Video": video_path.name,
            "Generations": output.response,
            "video_id": row["video_id"],
        }

        df = pd.DataFrame([row])
        save_to_results_files(df, tgt_file)
        torch.cuda.reset_peak_memory_stats()


def generate_timestamp() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


def generate_run_uuid() -> str:
    uid = str(uuid.uuid4())
    logger.info("Run ID: %s", uid)
    return uid


def save_to_results_files(df: pd.DataFrame, tgt_file: Path) -> None:
    df.to_json(tgt_file, orient="records", lines=True, mode="a")
    logger.info("added line to %s", tgt_file)


def batch_process_dataset(model: ModelInterface, posts_df: pd.DataFrame, tgt_file: str):
    assert isinstance(model, ModelvLLM_Benchmark), type(model)
    run_id = generate_run_uuid()
    reqs: Iterable[LlmReq] = list(to_iterof_llmreqs(posts_df))
    gen = model.llmlib_model.complete_batchof_reqs(batch=reqs)
    for response in tqdm(gen, desc="Processing dataset", total=len(posts_df)):
        data = response | {"run_id": run_id, "timestamp": generate_timestamp()}
        append_to_jsonl(tgt_file, data)


def append_to_jsonl(path: Path, row: dict) -> None:
    data_str = json.dumps(row, default=str, ensure_ascii=False)
    with open(path, "a") as f:
        f.write(data_str + "\n")


def run_benchmark(args: BenchmarkArgs) -> None:
    posts_df = get_posts_df(dataset_dir=args.dataset_dir)
    posts_df = posts_df.head(args.n_examples)
    if args.restart:
        posts_df = discard_posts_already_processed(posts_df, args.tgt_file)
    benchmark_videos(args, posts_df)


def discard_posts_already_processed(
    posts_df: pd.DataFrame, tgt_file: Path | str
) -> pd.DataFrame:
    results_df = read_results_file(file=tgt_file)
    success = results_df.query("success")
    logger.info("Already processed %d posts", success["video_id"].nunique())
    posts_df = posts_df[~posts_df["video_id"].isin(success["video_id"])]
    logger.info("%d posts remain to be processed", len(posts_df))
    return posts_df


def read_results_file(file: Path | str) -> pd.DataFrame:
    file = Path(file)
    if not file.exists():
        raise FileNotFoundError(file.absolute())
    return pd.read_json(file, orient="records", lines=True, dtype={"video_id": "str"})


@contextmanager
def profiler_context(no_op: bool = False) -> Generator[None, None, None]:
    if no_op:
        yield
        return

    profiler = Profiler(interval=0.01)
    try:
        with profiler:
            yield
    finally:
        os.makedirs("profiles", exist_ok=True)
        profiler.write_html(
            f"profiles/profile_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        )


def vllm_command(args: BenchmarkArgs) -> list[str]:
    return [
        "vllm",
        "serve",
        args.model_id,
        "--max-model-len=50000",
        "--max-seq-len-to-capture=50000",
        "--dtype=auto",
        f"--allowed-local-media-path={args.vllm_allowed_local_media_path}",
        "--limit-mm-per-prompt=image=50,video=1",
        "--disable-log-requests",
        f"--port={args.vllm_port}",
        "--host=127.0.0.1",
        "--disable-uvicorn-access-log",
        "--gpu-memory-utilization=0.95",
        f"--tensor-parallel-size={args.vllm_tensor_parallel_size}",
    ]


@contextmanager
def using_vllm_server(args: BenchmarkArgs) -> Generator[None, None, None]:
    cmd: list[str] = vllm_command(args)
    with spinup_vllm_server(no_op=not args.vllm_start_server, vllm_command=cmd):
        yield


if __name__ == "__main__":
    enable_info_logs()
    args = BenchmarkArgs()
    logger.info("%s", args)
    with profiler_context(no_op=not args.profile), using_vllm_server(args):
        run_benchmark(args)

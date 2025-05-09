from pathlib import Path
from bench_lib.utils import enable_info_logs, get_posts_df, to_dataset
from pydantic_settings import BaseSettings
from llmlib.vllm_model import dump_dataset_as_batch_request


class BenchmarkArgs(BaseSettings, cli_parse_args=True):
    model_id: str
    tgt_jsonl: Path
    n_examples: int = 10


if __name__ == "__main__":
    enable_info_logs()

    args = BenchmarkArgs()
    posts_df = get_posts_df()
    posts_df = posts_df.head(args.n_examples)
    dataset = to_dataset(posts_df)
    dump_dataset_as_batch_request(
        dataset=dataset,
        model_id=args.model_id,
        tgt_jsonl=Path(args.tgt_jsonl),
    )

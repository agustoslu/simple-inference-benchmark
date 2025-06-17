from pathlib import Path
from bench_lib.evaluation import benchmark_results_folder
import duckdb
import pandas as pd
from bench_lib.utils import Cols, enable_info_logs, get_answers_in_wide_format
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


def parse_and_dump_labels(folder: str, from_parquet: bool = False):
    if from_parquet:
        df = _read_parquet(folder)
    else:
        df = _read_jsonl(folder)

    wides = []
    unparsables = []
    for run_id, group_df in tqdm(list(df.groupby(Cols.run_id))):
        wide_df, unparsable = get_answers_in_wide_format(raw_jsonl_lines=group_df)
        wides.append(wide_df.assign(**{Cols.run_id: run_id}))
        unparsables.append(unparsable.assign(**{Cols.run_id: run_id}))
    wide_df = pd.concat(wides, ignore_index=True)
    unparsable = pd.concat(unparsables, ignore_index=True)

    print(
        "Folder: %s, Parsed: %d, Unparsable: %d"
        % (folder, len(wide_df), len(unparsable))
    )
    root_dir = benchmark_results_folder() / folder
    wide_df.to_csv(root_dir / "model_labels.csv", index=False)
    unparsable.to_csv(root_dir / "unparsable_rows.csv", index=False)


def _read_jsonl(folder: str) -> pd.DataFrame:
    root_dir = benchmark_results_folder() / folder
    jsonl_path = root_dir / "toxicainment_videos_log.jsonl"
    assert Path(jsonl_path).exists(), jsonl_path
    df = pd.read_json(jsonl_path, orient="records", lines=True)
    return df


def _read_parquet(folder: str) -> pd.DataFrame:
    root_dir = benchmark_results_folder() / folder
    con = duckdb.connect()
    df = con.from_parquet(str(root_dir / "*.parquet")).df()
    col_mapping = {
        "video_id": Cols.post_id,
        "run_id": Cols.run_id,
        "response": Cols.generations,
        "model": Cols.model_id,
    }
    df.rename(columns=col_mapping, inplace=True)
    df = drop_failed_rows(df)
    return df


def drop_failed_rows(df: pd.DataFrame) -> pd.DataFrame:
    failed = df.query("~success")
    if len(failed) > 0:
        logger.info("Dropping %d failed rows", len(failed))
        df = df.query("success")
    return df


if __name__ == "__main__":
    # models = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it", "MiniCPM-V-2.6"]
    # folders = [f"gemma-3-27b-it_{n:02d}" for n in range(3)]
    enable_info_logs()
    folders = ["vllm-qwen2.5-vl"]
    for folder in folders:
        parse_and_dump_labels(folder, from_parquet=True)

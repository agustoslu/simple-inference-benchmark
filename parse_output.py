from pathlib import Path
from bench_lib.evaluation import benchmark_results_folder
import pandas as pd
from bench_lib.utils import get_answers_in_wide_format
from tqdm import tqdm


def parse_and_dump_labels(folder: str):
    root_dir = benchmark_results_folder() / folder
    jsonl_path = root_dir / "toxicainment_videos_log.jsonl"
    assert Path(jsonl_path).exists(), jsonl_path
    df = pd.read_json(jsonl_path, orient="records", lines=True)

    wides = []
    unparsables = []
    for run_id, group_df in tqdm(list(df.groupby("Run_ID"))):
        wide_df, unparsable = get_answers_in_wide_format(raw_jsonl_lines=group_df)
        wides.append(wide_df)
        unparsables.append(unparsable)
    wide_df = pd.concat(wides, ignore_index=True)
    unparsable = pd.concat(unparsables, ignore_index=True)

    print(
        "Folder: %s, Parsed: %d, Unparsable: %d"
        % (folder, len(wide_df), len(unparsable))
    )
    wide_df.to_csv(root_dir / "model_labels.csv", index=False)
    unparsable.to_csv(root_dir / "unparsable_rows.csv", index=False)


if __name__ == "__main__":
    # models = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it", "MiniCPM-V-2.6"]
    # folders = [f"gemma-3-27b-it_{n:02d}" for n in range(3)]
    folders = ["gemini-2.0-flash-001"]
    for folder in folders:
        parse_and_dump_labels(folder)

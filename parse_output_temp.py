from pathlib import Path
from bench_lib.evaluation import benchmark_results_folder
import pandas as pd
from bench_lib.utils import get_answers_in_wide_format
from tqdm import tqdm
import glob
import os


def parse_and_dump_labels():
    root_dir = benchmark_results_folder()
    jsonl_paths = glob.glob(str(root_dir / "toxicainment_videos_log_Temp_*.jsonl"))
    

    for jsonl_path in jsonl_paths:
        assert Path(jsonl_path).exists(), jsonl_path
        df = pd.read_json(jsonl_path, orient="records", lines=True)
        for run_id, group_df in tqdm(list(df.groupby("Run_ID"))):
            wide_df, unparsable = get_answers_in_wide_format(raw_jsonl_lines=group_df)
            jsonl_path = Path(jsonl_path).stem
            model_id = group_df["Model ID"].iloc[0]
            folder_path = root_dir / jsonl_path / model_id
            if not folder_path.exists():
                os.makedirs(folder_path)
            print(
                "Folder: %s, Model ID: %s, Parsed: %d, Unparsable: %d"
                % (folder_path, model_id, len(wide_df), len(unparsable))
            )
            wide_df.to_csv(folder_path / "model_labels.csv", index=False)
            unparsable.to_csv(folder_path / "unparsable_rows.csv", index=False)


if __name__ == "__main__":
    parse_and_dump_labels()
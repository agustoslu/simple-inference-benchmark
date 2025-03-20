from pathlib import Path
from utils import get_answers_in_wide_format


def parse_and_dump_labels(model_id: str):
    root_dir = Path("results") / model_id
    jsonl_path = root_dir / "toxicainment_videos_log.jsonl"
    wide_df, unparsable = get_answers_in_wide_format(jsonl_path)
    print(
        "Model: %s, Parsed: %d, Unparsable: %d"
        % (model_id, len(wide_df), len(unparsable))
    )
    wide_df.to_csv(root_dir / "model_labels.csv", index=False)
    unparsable.to_csv(root_dir / "unparsable_rows.csv", index=False)


if __name__ == "__main__":
    models = ["gemma-3-4b-it", "gemma-3-12b-it", "gemma-3-27b-it", "MiniCPM-V-2.6"]
    for model_id in models:
        parse_and_dump_labels(model_id)

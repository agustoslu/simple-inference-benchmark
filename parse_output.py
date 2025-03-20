from pathlib import Path
from utils import get_answers_in_wide_format

root_dir = Path("results") / "gemma-3-4b-it"
jsonl_path = root_dir / "toxicainment_videos_log.jsonl"
wide_df, unparsable = get_answers_in_wide_format(jsonl_path)
print("Parsed", len(wide_df), "rows. And", len(unparsable), "rows were unparsable.")
wide_df.to_csv(root_dir / "model_labels.csv", index=False)

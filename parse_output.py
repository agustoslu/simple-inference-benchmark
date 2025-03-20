from pathlib import Path
from utils import get_answers_in_wide_format

root_dir = Path("results") / "gemma-3-27b-it"
jsonl_path = root_dir / "toxicainment_videos_log.jsonl"
wide_df = get_answers_in_wide_format(jsonl_path)
wide_df.to_csv(root_dir / "model_labels.csv", index=False)

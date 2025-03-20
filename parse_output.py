from utils import get_answers_in_wide_format

jsonl_path = "./results-gemma-3-12b-it/toxicainment_videos_log.jsonl"
wide_df = get_answers_in_wide_format(jsonl_path)
wide_df.to_csv("results-gemma-3-12b-it/model_labels.csv", index=False)

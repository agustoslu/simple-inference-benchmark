from utils import parse_output

# model_id = "openbmb/MiniCPM-V-2_6"
model_id = "openbmb/MiniCPM-V-2_6-int4"
model_labels_csv = f"model_labels_{model_id.replace('/', '_')}_03_17_25.csv"
# csv_file = "toxicainment_videos_log.jsonl"
csv_file = "toxicainment_videos_log_4bit.jsonl"
parse_output(csv_file, model_labels_csv, model_id)

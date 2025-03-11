from utils import parse_output

model_id = "openbmb/MiniCPM-V-2_6"
model_labels_csv = f"model_labels_{model_id.replace('/', '_')}.csv"
csv_file = "toxicainment_videos_log.csv"
parse_output(csv_file, model_labels_csv, model_id)

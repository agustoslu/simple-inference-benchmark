import os
from pyaml_env import parse_config

config = parse_config("./config.yaml")
output_folder = "./simple-inference-benchmark/dataset/FineVideo_20_Samples"
os.makedirs(output_folder, exist_ok=True)

base_url = "https://huggingface.co/datasets/HuggingFaceFV/finevideo/resolve/main/data/"

template = "train-{i:05d}-of-01357.parquet"

access_token = config["hf_token"]


def download_finevideo_parquet(i: int) -> None:
    """i is the split index, it ranges from 0 to 1357"""
    file_name = template.format(i=i)
    file_url = f"{base_url}{file_name}"
    header = f'--header="Authorization: Bearer {access_token}"'
    print(f"Downloading {file_name}...")
    os.system(f"wget {header} -P {output_folder} {file_url}")


download_finevideo_parquet(i=0)

print("Done downloading")

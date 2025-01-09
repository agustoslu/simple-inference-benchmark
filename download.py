import os
from pathlib import Path
from pyaml_env import parse_config

config = parse_config("./config.yaml")


base_dir = Path(__file__).parent / "dataset"
DATASET_PATH = base_dir / "FineVideo_parquet"
MP4_DATASET_PATH = base_dir / "FineVideo_mp4"
DATASET_PATH.mkdir(parents=True, exist_ok=True)
MP4_DATASET_PATH.mkdir(parents=True, exist_ok=True)

base_url = "https://huggingface.co/datasets/HuggingFaceFV/finevideo/resolve/main/data/"

template = "train-{i:05d}-of-01357.parquet"

access_token = config["hf_token"]


def download_finevideo_parquet(i: int) -> None:
    """i is the split index, it ranges from 0 to 1357"""
    file_name = template.format(i=i)
    file_url = f"{base_url}{file_name}"
    header = f'--header="Authorization: Bearer {access_token}"'
    print(f"Downloading {file_name}...")
    os.system(f"wget {header} -P {DATASET_PATH} {file_url}")


if __name__ == "__main__":
    download_finevideo_parquet(i=0)
    print("Done downloading")

# when connected to server downloaded files might not be shown in the folder on the explorer panel
# but still simple ls check will show that they are there
import os

output_folder = "./simple-inference-benchmark/dataset/FineVideo_20_Samples"
os.makedirs(output_folder, exist_ok=True)

base_url = "https://huggingface.co/datasets/HuggingFaceFV/finevideo/resolve/main/data/"

file_names = [
    "train-00000-of-01357.parquet",
    "train-00001-of-01357.parquet",
    "train-00002-of-01357.parquet",
    "train-00003-of-01357.parquet",
    "train-00004-of-01357.parquet",
    "train-00005-of-01357.parquet", 
    "train-00006-of-01357.parquet",
    "train-00007-of-01357.parquet", 
    "train-00008-of-01357.parquet",
    "train-00009-of-01357.parquet",
    "train-00011-of-01357.parquet",
    "train-00012-of-01357.parquet",
    "train-00013-of-01357.parquet",
    "train-00014-of-01357.parquet",
    "train-00015-of-01357.parquet",
    "train-00016-of-01357.parquet",
    "train-00017-of-01357.parquet",
    "train-00018-of-01357.parquet",
    "train-00019-of-01357.parquet",
    "train-00020-of-01357.parquet",
]


access_token = ""

for file_name in file_names:
    file_url = f"{base_url}{file_name}"
    header = f'--header="Authorization: Bearer {access_token}"'
    
    print(f"Downloading {file_name}...")
    os.system(f"wget {header} -P {output_folder} {file_url}")

print("All videos are downloaded")

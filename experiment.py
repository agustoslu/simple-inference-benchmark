import os
import time
import random
import yaml
import cv2
import json
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer
from run_minicpm import extract_videos_from_parquet, benchmark_videos, process_video

with open("config.yaml", "r") as config_file:
    config = yaml.safe_load(config_file)

# Experiment with 5 videos using the settings in the config file
video_paths = extract_videos_from_parquet(
    "./simple-inference-benchmark/dataset/FineVideo_20_Samples", 
    "./simple-inference-benchmark/dataset/temp_videos", 
    num_videos=config['num_videos'],
    seed=config['seed']
)

for fps in config['fps_settings']:
    for token_limit in config['token_settings']:
        print(f"\nRunning experiment with FPS={fps}, Token Limit={token_limit}...\n")
        
        benchmark_results = benchmark_videos(
            video_paths, 
            seconds_per_frame=fps,
            token_limit=token_limit,
            num_samples=config["num_samples"], 
            hf_token=config["hf_token"],
            compile=config["compile"] 
        )
        
        print(f"Completed experiment for FPS={fps}, Token Limit={token_limit}")

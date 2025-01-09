from run_minicpm import benchmark_videos, sample_n_videos
from pyaml_env import parse_config


config = parse_config("./config.yaml")


# Experiment with 5 videos using the settings in the config file
video_paths = sample_n_videos(
    n=config["num_videos"],
    seed=config["seed"],
)

for fps in config["fps_settings"]:
    for token_limit in config["token_settings"]:
        print(f"\nRunning experiment with FPS={fps}, Token Limit={token_limit}...\n")

        benchmark_results = benchmark_videos(
            video_paths,
            token_limit=token_limit,
            num_samples=config["num_samples"],
            hf_token=config["hf_token"],
            compile=config["compile"],
        )

        print(f"Completed experiment for FPS={fps}, Token Limit={token_limit}")

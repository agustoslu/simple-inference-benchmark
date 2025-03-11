import math
from pathlib import Path
import pandas as pd
from ast import literal_eval
from collections import defaultdict
import os
from decord import VideoReader, cpu
from PIL import Image
import json
import re


def toxicainment_data_folder() -> Path:
    dss_home = os.environ["DSS_HOME"]
    return Path(dss_home) / "toxicainment"

def get_posts():
    videos = {}
    slides = defaultdict(list)
    folder = toxicainment_data_folder() / "2025-02-07-saxony-labeled-data"
    media_dir = folder / "media"
    posts_df = pd.read_csv(folder / "media_metadata.csv")
    posts_df["filenames"] = (
        posts_df["filenames"].str.replace("\n", ",").apply(literal_eval)
    )  

    for idx, row in posts_df.iterrows():
        filenames: list[str] = row["filenames"]
        is_video: bool = filenames[0].endswith(".mp4")

        if is_video:
            video_path = media_dir / filenames[0]
            assert video_path.exists()
            videos[idx] = {
                "video_path": str(video_path),
                "author_name": row.get("author_name", "NA"),
                "video_description": row.get("video_description", "")
            }
        
        else:
            has_audio_file = isinstance(row.get("audio_file"), str)
            pic_paths = [media_dir / f for f in filenames]
            pic_path_list = [str(pic_path) for pic_path in pic_paths]  

            slides[idx] = {
                "slide_path": pic_path_list,
                "author_name": row.get("author_name", "NA"),
                "slide_description": row.get("slide_description", "")
            }

            if has_audio_file:
                audio_path = media_dir / row["audio_file"]
                assert audio_path.exists()
                slides[idx]["audio_file"] = str(audio_path)
            
    return videos, slides

def get_labels():
    media_dir = Path(__file__).parent / "media"

    human_labels = media_dir / "human_labels.csv"
    model_labels = media_dir / "model_labels.csv"
    merged = media_dir / "merged_labels.csv"

    human_df = pd.read_csv(human_labels)
    model_df = pd.read_csv(model_labels)

    human_cols = ['post_id', 'author', 'classification_by', 'is_political', 'is_political_comment', 'is_saxony_election', 'is_saxony_election_comment', 'is_intolerant', 'is_intolerant_comment', 'is_hedonic_entertainment', 'is_hedonic_entertainment_comment', 'is_eudaimonic_entertainment', 'is_eudaimonic_entertainment_comment']
    model_cols = ['post_id', 'author', 'classification_by', 'is_political', 'is_political_comment', 'is_saxony_election', 'is_saxony_election_comment', 'is_intolerant', 'is_intolerant_comment', 'is_hedonic_entertainment', 'is_hedonic_entertainment_comment', 'is_eudaimonic_entertainment', 'is_eudaimonic_entertainment_comment']

    human_df = [human_cols]
    model_df = [model_cols]
    merged_df = pd.merge(human_df, on="post_id", how="outer")
    merged_df.to_csv(merged, index=False)
    print(f"saved to {merged}")

    return merged_df

def video_to_frames(video_path: Path):
    vr = VideoReader(str(video_path), ctx=cpu(0))
    frame_indices = compute_frame_indices(
        vid_n_frames=len(vr), 
        vid_fps=vr.get_avg_fps(), 
        max_n_frames=200
    )    
    frames = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(frame.astype("uint8")) for frame in frames]
    return frames

# Parsing output

def extract_json(text):
    matches = re.findall(r'```json\n(.*?)```', text, re.DOTALL)
    return [json.loads(m) for m in matches if m.strip()]

def get_post_id(post_id):
    return post_id.split('/')[-1].replace('.mp4', '').strip(';')

def parse_output(log_path, model_labels_csv, model_id):
    df = pd.read_json(log_path, orient='records', lines=True)
    model_labels = []
      
    videos, _ = get_posts()
    
    video_meta = {Path(v["video_path"]).name.replace(".mp4", ""): v["author_name"] for v in videos.values()}

    for i in df.index:
        raw_post_id = df.loc[i, "Processed_Video"]
        generation = df.loc[i, "Generations"]
        
        post_id = get_post_id(raw_post_id)
        author_name = video_meta.get(post_id, "NA")
        
        json_blocks = extract_json(generation)
        if not json_blocks:
            print(f"Skipping post_id {post_id}: No valid JSON found.")
            continue
        
        answer_block = json_blocks[0]
        answers_dict = {}
        for item in answer_block.get("answers", []):
            question = item["question"]
            answers_dict[f"{question}"] = item["answer"]
            answers_dict[f"{question}_comment"] = item.get("comment", "")
        model_labels.append({"post_id": post_id, "author": author_name, "classification_by": model_id, **answers_dict})
        
    pd.DataFrame(model_labels).to_csv(model_labels_csv, index=False)
    print(f"Model labels saved to {model_labels_csv}")
    return model_labels_csv


def compute_frame_indices(vid_n_frames: int, vid_fps: float, max_n_frames: int):
    """
    This function will return the frames starting at 0 every second.
    Unless that number exceeds max_n_frames, in which case it will return max_n_frames frames evenly spaced out, starting at 0.
    """
    assert isinstance(vid_n_frames, int), vid_n_frames
    assert isinstance(max_n_frames, int), max_n_frames
    vid_fps = int(vid_fps)
    fps_n_frames = math.ceil(vid_n_frames / vid_fps)
    if fps_n_frames <= max_n_frames:
        return list(range(0, vid_n_frames - 1, vid_fps))
    else:
        return list(range(0, vid_n_frames - 1, vid_n_frames // max_n_frames))
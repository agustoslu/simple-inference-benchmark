from pathlib import Path
import pandas as pd
from ast import literal_eval
from collections import defaultdict
import os
from decord import VideoReader, cpu
from PIL import Image


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
    MAX_NUM_FRAMES = 64
    vr = VideoReader(str(video_path), ctx=cpu(0))
    total = []
    total_frames = len(vr)
    for i in vr:
        total.append(i)
        
    print(f"Total Frames: {total_frames}")
    fps = vr.get_avg_fps()
    print(f"FPS: {fps}")
    total_frames_int = int(total_frames)
    print(f"total transformed: {total_frames_int}")
    if total_frames <= 6000:
        frame_indices = fps_sample(int(total_frames), round(fps), range(total_frames))
        print(f"frame indices: {frame_indices}")
        print(f"total frames passed(fps): {len(frame_indices)}")

    elif total_frames > 6000:
        total_fps = total[:6000]
        total_uni = total[6000:]
            # total uni starts from zero since it's newly created even tough frames after 4000 are added
        frame_indices = fps_sample(int(len(total_fps)), round(fps), range(len(total_fps)))
        frame_indices_uni = uniform_sample(range(len(total_uni)), MAX_NUM_FRAMES)
        print(f"frame indices_fps: {frame_indices}")
        print(f"total frames passed(fps): {len(frame_indices)}")
        print(f"frame indices_uni: {frame_indices_uni}")
        print(f"total frames passed(uni): {len(frame_indices_uni)}")
        
    frames = vr.get_batch(frame_indices).asnumpy()
    frames = [Image.fromarray(frame.astype("uint8")) for frame in frames]
    return frames, total_frames

# Employ uniform sampling for frames
def uniform_sample(xs, n):
    gap = len(xs) / n
    idxs = [int(i * gap + gap / 2) for i in range(n)]
    return [xs[i] for i in idxs]

def fps_sample(xs, fps, total_range): 
    idxs = [i * fps for i in range(xs // fps)]
    return [total_range[i] for i in idxs]
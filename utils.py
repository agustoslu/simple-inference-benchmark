from pathlib import Path
import pandas as pd
from ast import literal_eval
from collections import defaultdict
import os


def toxicainment_data_folder() -> Path:
    dss_home = os.environ["DSS_HOME"]
    return Path(dss_home) / "toxicainment"

from PIL import Image
#import cv2

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
    
    # s = ' '.join(slides[299]['slide_path'])
    # #print(s)
    # import regex as re
    # #s2 = s.strip()

    # s2 = re.sub(r'(?<=\.(jpg|png|jpeg|gif|bmp|tiff))\s+(?=/)', '', s, flags=re.IGNORECASE)
    # # how to batch mp3 too?


    #print(s2)
    
    return videos, slides


def batch_slides(slides):
    batched_slides = {}

    for idx, row in slides.items():
        pic_path_list = row.get("slide_path", [])
        slide_imgs = []

        for pic in pic_path_list:
            try:
                image = Image.open(pic).convert("RGB")
                slide_imgs.append(image)
            except Exception as e:
                print(f"Error loading image {pic}: {e}")
        
        batched_slides[idx] = {
            "slide_path": pic_path_list,
            "author_name": row.get("author_name", "NA"),
            "slide_description": row.get("slide_description", ""),
            "frames": slide_imgs
        }
    return batched_slides



# if __name__ == "__main__":
#     videos, slides = get_posts()
#     batch_slides(slides)

# def get_labels():
#     media_dir = Path(__file__).parent / "media"

#     human_labels = media_dir / "human_labels.csv"
#     model_labels = media_dir / "model_labels.csv"
#     merged = media_dir / "merged_labels.csv"

#     human_df = pd.read_csv(human_labels)
#     model_df = pd.read_csv(model_labels)

#     human_cols = ['post_id', 'author', 'classification_by', 'is_political', 'is_political_comment', 'is_saxony_election', 'is_saxony_election_comment', 'is_intolerant', 'is_intolerant_comment', 'is_hedonic_entertainment', 'is_hedonic_entertainment_comment', 'is_eudaimonic_entertainment', 'is_eudaimonic_entertainment_comment']
#     model_cols = ['post_id', 'author', 'classification_by', 'is_political', 'is_political_comment', 'is_saxony_election', 'is_saxony_election_comment', 'is_intolerant', 'is_intolerant_comment', 'is_hedonic_entertainment', 'is_hedonic_entertainment_comment', 'is_eudaimonic_entertainment', 'is_eudaimonic_entertainment_comment']

#     human_df = [human_cols]
#     model_df = [model_cols]
#     merged_df = pd.merge(human_df, on="post_id", how="outer")
#     merged_df.to_csv(merged, index=False)
#     print(f"saved to {merged}")

#     return merged_df

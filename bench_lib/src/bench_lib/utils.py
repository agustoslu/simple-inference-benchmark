from pathlib import Path
from typing import Iterable
from json_repair import json_repair
import pandas as pd
from ast import literal_eval
import os
import json
import re
import logging
from llmlib.base_llm import Conversation, Message, LlmReq
from llmlib.huggingface_inference import is_video
from functools import cache
import subprocess

logger = logging.getLogger(__name__)


def dataset_files(dataset_dir: Path) -> tuple[Path, Path]:
    media_dir = dataset_dir / "media"
    assert_exists(media_dir)

    csv = dataset_dir / "media_metadata.csv"
    if csv.exists():
        metadata_file = csv
        posts_df = pd.read_csv(metadata_file, dtype={"video_id": "str"})
        return media_dir, posts_df

    metadata_file = dataset_dir / "ldf.parquet"
    assert_exists(metadata_file)
    posts_df = pd.read_parquet(metadata_file)
    posts_df["video_id"] = posts_df["video_id"].astype("str")
    posts_df["filenames"] = posts_df["filenames"].astype("str")
    return media_dir, posts_df


def assert_exists(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(path.absolute())


def saxony_dataset_dir() -> Path:
    dss_home = os.environ["DSS_HOME"]
    return Path(dss_home) / "toxicainment" / "2025-02-07-saxony-labeled-data"


def get_posts_df(dataset_dir: Path) -> pd.DataFrame:
    media_dir, posts_df = dataset_files(dataset_dir)
    posts_df["filenames"] = (
        posts_df["filenames"].str.replace("\n", ",").apply(literal_eval)
    )
    posts_df = posts_df.assign(
        is_video=posts_df["filenames"].apply(lambda x: is_video(x[0]))
    )
    logger.info(
        "Num Posts: %d. Of those, %d are videos and %d are slides.",
        len(posts_df),
        len(posts_df.query("is_video")),
        len(posts_df.query("~is_video")),
    )
    posts_df = posts_df.assign(
        filenames=posts_df["filenames"].apply(lambda x: [media_dir / f for f in x]),
        video_path=posts_df["filenames"].apply(lambda x: x[0]),
    )
    desired_cols = [
        "video_id",
        "video_path",
        "author_name",
        "video_description",
        "filenames",
        "is_video",
    ]
    return posts_df[desired_cols]


@cache
def read_prompt_template() -> str:
    with open(Path(__file__).parent / "prompts" / "prompt.txt", "r") as f:
        text: str = f.read()
    return text


def fill_prompt(row_dict: dict, template: str) -> str:
    author = row_dict["author_name"]
    description = row_dict["video_description"]
    filled = template % (author, description)
    return filled


def to_dataset(posts_df: pd.DataFrame) -> Iterable[Conversation]:
    for _, row in posts_df.iterrows():
        convo = to_convo(row.to_dict())
        yield convo


def to_convo(posts_df_row: dict) -> Conversation:
    template = read_prompt_template()
    filled_prompt = fill_prompt(row_dict=posts_df_row, template=template)
    convo = [Message(role="user", msg=filled_prompt, files=posts_df_row["filenames"])]
    return convo


def to_iterof_llmreqs(posts_df: pd.DataFrame) -> Iterable[LlmReq]:
    for _, row in posts_df.iterrows():
        row_dict = row.to_dict()
        yield LlmReq(
            convo=to_convo(row_dict),
            metadata={"video_id": row_dict["video_id"]},
        )


def get_labels():
    media_dir = Path(__file__).parent / "media"

    human_labels = media_dir / "human_labels.csv"
    model_labels = media_dir / "model_labels.csv"
    merged = media_dir / "merged_labels.csv"

    human_df = pd.read_csv(human_labels)
    model_df = pd.read_csv(model_labels)

    human_cols = [
        "post_id",
        "author",
        "classification_by",
        "is_political",
        "is_political_comment",
        "is_saxony_election",
        "is_saxony_election_comment",
        "is_intolerant",
        "is_intolerant_comment",
        "is_hedonic_entertainment",
        "is_hedonic_entertainment_comment",
        "is_eudaimonic_entertainment",
        "is_eudaimonic_entertainment_comment",
    ]
    model_cols = [
        "post_id",
        "author",
        "classification_by",
        "is_political",
        "is_political_comment",
        "is_saxony_election",
        "is_saxony_election_comment",
        "is_intolerant",
        "is_intolerant_comment",
        "is_hedonic_entertainment",
        "is_hedonic_entertainment_comment",
        "is_eudaimonic_entertainment",
        "is_eudaimonic_entertainment_comment",
    ]

    human_df = [human_cols]
    model_df = [model_cols]
    merged_df = pd.merge(human_df, on="post_id", how="outer")
    merged_df.to_csv(merged, index=False)
    print(f"saved to {merged}")

    return merged_df


# Parsing output


def get_post_id(post_id):
    return post_id.split("/")[-1].replace(".mp4", "").strip(";")


def clean_json_string(json_string):
    # json strings include markdown chars and that results in JSON decoding error if not cleaned
    cleaned = re.sub(r"^```json\n?|```$", "", json_string.strip(), flags=re.MULTILINE)
    return cleaned


def parse_output(log_path, model_labels_csv, model_id, dataset_dir: Path):
    df = pd.read_json(log_path, orient="records", lines=True)
    model_labels = []
    unparsable_videos = []

    posts_df = get_posts_df(dataset_dir)
    video_meta = {
        Path(v["video_path"]).name.replace(".mp4", ""): v["author_name"]
        for _, v in posts_df.iterrows()
    }

    pattern = re.compile(
        r"(intolerant|intolerance|political|saxony|hedonic|eudaimonic)[:\s]*(Yes|No|1|0)",
        re.IGNORECASE,
    )

    for i in df.index:
        raw_post_id = df.loc[i, "Processed_Video"]
        generation = df.loc[i, "Generations"]

        post_id = get_post_id(raw_post_id)
        author_name = video_meta.get(post_id, "NA")

        if not generation:
            print(f"Skipping post_id {post_id}: No valid data found.")
            unparsable_videos.append(post_id)
            continue

        try:
            cleaned_generation = (
                clean_json_string(generation)
                if isinstance(generation, str)
                else generation
            )
            generation_data = (
                json.loads(cleaned_generation)
                if isinstance(cleaned_generation, str)
                else generation
            )
        except json.JSONDecodeError:
            print(f"Skipping post_id {post_id}: JSON decoding error.")
            unparsable_videos.append(post_id)
            continue

        answers_dict = {}

        for item in generation_data.get("answers", []):
            question = item["question"].lower()
            response = str(item["answer"]).strip()

            match = pattern.search(response)
            if match:
                category, label = match.groups()
                label = "Yes" if label in ["Yes", "1"] else "No"
                answers_dict[f"{category}"] = label
            else:
                answers_dict[question] = response
                answers_dict[f"{question}_comment"] = item.get("comment", "")

        model_labels.append(
            {
                "post_id": post_id,
                "author": author_name,
                "classification_by": model_id,
                **answers_dict,
            }
        )

    pd.DataFrame(model_labels).to_csv(model_labels_csv, index=False)
    print(f"Model labels saved to {model_labels_csv}")
    print(f"Unparsable instances: {len(unparsable_videos)} out of 212 videos.")
    return model_labels_csv


def get_answers_in_wide_format(
    raw_jsonl_lines: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = raw_jsonl_lines
    if "post_id" not in df.columns:
        assert df["Processed_Video"].str.contains(".mp4$").all(), (
            "The line below is for videos"
        )
        df["post_id"] = df["Processed_Video"].str[-23:-4]
    assert df["Run_ID"].nunique() == 1, "There should be only one run id"
    dfs = []
    unparsable = []
    for i, row in df.iterrows():
        try:
            gens = row[Cols.generations]
            if isinstance(gens, list):
                gens = gens[0]
            assert isinstance(gens, str), f"Generations must be a str, not {type(gens)}"
            answers = json_repair.loads(gens)["answers"]
            if answers == "":
                unparsable.append(row)
                continue

            extra_data = {
                Cols.run_id: row[Cols.run_id],
                Cols.model_id: row[Cols.model_id],
                Cols.post_id: row[Cols.post_id],
            }
            dfs.append(pd.DataFrame(answers).assign(**extra_data))
        except (json.JSONDecodeError, TypeError, KeyError, AttributeError) as e:
            unparsable.append(row)

    answers_long = pd.concat(dfs, ignore_index=True)
    # drop duplicated answers to questions: [{is_intolerance, yes}, {is_intolerance, no}, {is_intolerance, ...}]
    answers_long = answers_long.groupby(
        [Cols.post_id, "question"], as_index=False
    ).last()

    idx_cols = [Cols.run_id, Cols.model_id, Cols.post_id]
    answers_wide = answers_long.pivot(
        index=idx_cols,
        columns="question",
        values="answer",
    ).reset_index()
    comments_wide = answers_long.pivot(
        index=idx_cols,
        columns="question",
        values="comment",
    ).reset_index()
    comments_wide.columns = [
        *idx_cols,
        *(c + "_comment" for c in answers_wide.columns if c not in idx_cols),
    ]
    wide = pd.merge(answers_wide, comments_wide, on=idx_cols)
    wide = fix_column_typos(wide)
    return wide, pd.DataFrame(unparsable)


def fix_column_typos(df: pd.DataFrame) -> pd.DataFrame:
    from_to = {
        "is_eudaimonic_entartainment": "is_eudaimonic_entertainment",
        "is_eudaimonic_entartainment_comment": "is_eudaimonic_entertainment_comment",
    }
    from_to = {k: v for k, v in from_to.items() if k in df.columns}
    df = df.rename(columns=from_to)
    return df


def mute_video(dataset_dir: Path, input_path: Path) -> Path:
    output_path = Path(dataset_dir / f"{input_path.stem}_muted.mp4")
    assert is_video(input_path), f"Input path {input_path.stem} is not a video file"
    if not output_path.exists():
        cmd = [
            "ffmpeg",
            "-y",
            "-i",
            str(input_path),
            "-an",  # remove audio
            "-c:v",
            "copy",
            str(output_path),
        ]
        subprocess.run(
            cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    return output_path


@cache
def load_transcript_df(dataset_dir: Path) -> pd.DataFrame:
    transcript_dir = dataset_dir / "transcripts"
    assert_exists(transcript_dir)
    csv = transcript_dir / "whisper_transcriptions.csv"
    if not csv.exists():
        raise FileNotFoundError(csv)
    df = pd.read_csv(csv)
    df["video_id"] = df["video_id"].astype(str)
    df = df.set_index("video_id")
    return df


def get_transcript(dataset_dir: Path, video_id: str) -> list[dict]:
    df = load_transcript_df(dataset_dir)
    video_id = str(video_id)
    if video_id not in df.index:
        raise KeyError(f"Transcript for video_id {video_id} not found.")
    row = df.loc[video_id]
    return row["chunks"]


class Cols:
    post_id = "post_id"
    run_id = "Run_ID"
    model_id = "Model ID"
    generations = "Generations"


def enable_info_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

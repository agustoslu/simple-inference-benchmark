from pathlib import Path
import pandas as pd
from ast import literal_eval
import os
import json
import re
import logging

logger = logging.getLogger(__name__)


def toxicainment_data_folder() -> Path:
    dss_home = os.environ["DSS_HOME"]
    return Path(dss_home) / "toxicainment"


def get_posts_df() -> pd.DataFrame:
    folder = toxicainment_data_folder() / "2025-02-07-saxony-labeled-data"
    media_dir = folder / "media"
    posts_df = pd.read_csv(folder / "media_metadata.csv", dtype={"video_id": str})
    posts_df["filenames"] = (
        posts_df["filenames"].str.replace("\n", ",").apply(literal_eval)
    )
    posts_df = posts_df.assign(
        is_video=posts_df["filenames"].apply(lambda x: x[0].endswith(".mp4"))
    )
    logger.info(
        "Dropping %d non-video posts out of %d. Keeping %d posts.",
        len(posts_df.query("~is_video")),
        len(posts_df),
        len(posts_df.query("is_video")),
    )
    posts_df = posts_df.query("is_video")
    posts_df = posts_df.assign(
        video_path=posts_df["filenames"].apply(lambda x: media_dir / x[0])
    )
    desired_cols = ["video_id", "video_path", "author_name", "video_description"]
    return posts_df[desired_cols]


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


def parse_output(log_path, model_labels_csv, model_id):
    df = pd.read_json(log_path, orient="records", lines=True)
    model_labels = []
    unparsable_videos = []

    posts_df = get_posts_df()
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


def get_answers_in_wide_format(raw_jsonl_lines: pd.DataFrame) -> pd.DataFrame:
    df = raw_jsonl_lines
    assert df["Processed_Video"].str.contains(".mp4$").all(), (
        "The line below is for videos"
    )
    df["post_id"] = df["Processed_Video"].str[-23:-4]
    assert df["Run_ID"].nunique() == 1, "There should be only one run id"
    if isinstance(df["Generations"].values[0], list):
        df = df.assign(Generations=df["Generations"].apply(lambda x: x[0]))
    answers_by_post = df["Generations"].str.extract(r"```json(.*)```", flags=re.DOTALL)[
        0
    ]
    dfs = []
    unparsable = []
    for i, row in df.iterrows():
        try:
            answers = json.loads(answers_by_post[i])["answers"]
            extra_data = {
                Cols.run_id: row[Cols.run_id],
                Cols.model_id: row[Cols.model_id],
                Cols.post_id: row[Cols.post_id],
            }
            dfs.append(pd.DataFrame(answers).assign(**extra_data))
        except (json.JSONDecodeError, TypeError) as e:
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


class Cols:
    post_id = "post_id"
    run_id = "Run_ID"
    model_id = "Model ID"


def enable_info_logs() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

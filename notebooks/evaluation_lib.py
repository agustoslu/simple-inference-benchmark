import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def visualize_runtime(df: pd.DataFrame) -> None:
    # The model takes longer for more input frames and for more output tokens generated
    fig = plt.figure(figsize=(6, 4))
    ax = fig.add_subplot(111)
    scatter = ax.scatter(
        df["Tokens_Generated"], df["Model_Runtime"], c=df["Total_Frames"]
    )
    plt.colorbar(scatter, label="Input Frames Per Video")
    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("Output Tokens Generated")
    ax.set_ylabel("Model Runtime")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    return fig


def plot_question_hists(
    hist_data: dict[str, tuple[np.ndarray, np.ndarray]], title: str
):
    fig, axes = plt.subplots(1, 5, figsize=(16, 2.5))
    fig.tight_layout(pad=3.0)
    fig.suptitle(title)

    for ax, (question, (x_vals, y_vals)) in zip(axes, hist_data.items()):
        ax.bar(x_vals, y_vals, width=0.05)
        ax.set_title(question)
        ax.set_ylabel("Fraction")
        ax.set_xlabel("No=0 or Yes=1")
        ax.set_ylim(0, 1.1)
        ax.set_xlim(-0.1, 1.1)
        ax.grid(alpha=0.5)


def performance_by_category(labels_wide, ref_wide, questions: list[str]):
    """Both dfs are in wide format"""
    assert set(["post_id", *questions]).issubset(labels_wide.columns)
    joined = join_wides(labels_wide, ref_wide, questions)
    return joined.groupby("variable", as_index=False)["is_correct"].mean()


def join_wides(labels_wide, ref_wide, questions: list[str]):
    joined = pd.merge(
        pd.melt(ref_wide, id_vars="post_id", value_vars=questions),
        pd.melt(labels_wide, id_vars="post_id", value_vars=questions),
        on=["post_id", "variable"],
        suffixes=("_pred", "_ref"),
        how="inner",
    ).assign(is_correct=lambda df: df["value_pred"] == df["value_ref"])

    return joined


def to_binary_series(s: pd.Series):
    assert s.isin(["yes", "no"]).all()
    return s.map({"yes": 1, "no": 0})


def get_means(labels_df: pd.DataFrame, question: str):
    labels_df = labels_df.assign(question_binary=to_binary_series(labels_df[question]))
    return labels_df.groupby("post_id")["question_binary"].mean()


def load_ai_labels(
    models: list[str], questions: list[str], comment_cols: list[str]
) -> pd.DataFrame:
    all_ai_labels = []
    for model_id in models:
        ai_labels = pd.read_csv(
            f"../results/{model_id}/model_labels.csv", dtype={"post_id": str}
        )
        ai_labels = ai_labels[["post_id", *questions, *comment_cols]]
        rows_w_na = ai_labels[questions].isna().any(axis=1)
        print(f"{model_id}: filtering {rows_w_na.sum()} rows with NA values")
        complete_ai_labels = ai_labels[~rows_w_na]
        print(f"{model_id}: {len(complete_ai_labels)} rows after filtering")
        all_ai_labels.append(complete_ai_labels.assign(model_id=model_id))

    all_ai_labels = pd.concat(all_ai_labels)
    return all_ai_labels


def difficulty_score(df: pd.DataFrame):
    difficulty = (
        (1 - ((df - 0.5).abs() / 0.5)).mean(axis=1).sort_values(ascending=False)
    )
    return difficulty

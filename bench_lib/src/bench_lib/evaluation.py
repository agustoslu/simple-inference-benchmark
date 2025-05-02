import os
from pathlib import Path
from typing import Any, Iterable
import krippendorff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bench_lib.utils import Cols
from sklearn.metrics import f1_score, precision_score, recall_score
import logging
import seaborn as sns
from bert_score import score


logger = logging.getLogger(__name__)


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


def performance_by_category(labels_long, ref_long):
    """Both dfs are in long format. But the ref might be smaller than the labels."""
    assert set(["value", "variable", Cols.post_id]).issubset(labels_long.columns)
    assert set(["value", "variable", Cols.post_id]).issubset(ref_long.columns)
    comparison = pd.merge(
        ref_long, labels_long, on=[Cols.post_id, "variable"], suffixes=("_ref", "_pred")
    )
    df = comparison.groupby("variable", as_index=False).apply(compute_metrics)
    return df


def compute_metrics(x):
    y_true = x["value_ref"]
    y_pred = x["value_pred"]
    accuracy = (y_true == y_pred).mean()
    f1 = f1_score(y_true=y_true, y_pred=y_pred)
    precision = precision_score(y_true=y_true, y_pred=y_pred)
    recall = recall_score(y_true=y_true, y_pred=y_pred)
    return pd.Series(
        {
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "n_samples": len(y_true),
        }
    )


def compute_ai_perfs(
    human_labels: pd.DataFrame, ai_labels: pd.DataFrame, questions: list[str]
) -> pd.DataFrame:
    """both input dfs are in long format"""
    df = ai_labels.groupby(Cols.model_id).apply(
        performance_by_category,
        ref_long=human_labels,
    )
    df = df.reset_index().drop(columns=["level_1"])
    return df


def plot_ai_perfs(ai_perfs, order: list[str], x_order: list[str], y: str):
    fig, ax = plt.subplots(figsize=(9, 4))
    sns.barplot(
        data=ai_perfs,
        x="variable",
        y=y,
        hue=Cols.model_id,
        hue_order=order,
        order=x_order,
        palette="viridis",
        ax=ax,
    )
    ax.set_xlabel("")  # Hide xlabel
    ax.set_ylabel(y.capitalize())
    ax.set_xticks(ax.get_xticks())
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")
    # Move legend outside to the right
    ax.legend(
        title=Cols.model_id,
        bbox_to_anchor=(1.0, 1),
        loc="upper left",
    )
    ax.grid(alpha=0.5)
    plt.tight_layout()
    plt.close()  # Close the figure to prevent double display
    return fig


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
    folders: list[str],
    questions: list[str],
    comment_cols: list[str],
) -> pd.DataFrame:
    all_longs = []
    for folder in folders:
        fpath = model_label_fpath(folder)
        assert fpath.exists(), f"File {fpath} does not exist"
        ai_labels = pd.read_csv(fpath, dtype={"post_id": str})
        ai_labels = ai_labels[
            [Cols.run_id, Cols.model_id, Cols.post_id, *questions, *comment_cols]
        ]
        long = ai_labels_wide_to_long(ai_labels, questions, comment_cols)

        na_rows = long.query("value.isna()")
        if len(na_rows) > 0:
            logger.info("Removing %d rows with NA values", len(na_rows))
            long.query("value.notna()", inplace=True)

        long = long.assign(
            value=lambda df: df["value"].map({"yes": 1, "no": 0, "0": 0, "1": 1})
        )
        not_yesno_rows = long.query("~value.isin((0, 1))")
        if len(not_yesno_rows) > 0:
            logger.info(
                "Removing %d rows with answer different from yes/no values. E.g. '%s'",
                len(not_yesno_rows),
                not_yesno_rows["value"].values[0],
            )
            long.query("value.isin([0, 1])", inplace=True)

        all_longs.append(long)

    all_longs = pd.concat(all_longs)
    return all_longs


def ai_labels_wide_to_long(
    ai_labels: pd.DataFrame, questions: list[str], comment_cols: list[str]
) -> pd.DataFrame:
    id_vars = [Cols.post_id, Cols.run_id, Cols.model_id]
    ans = ai_labels.melt(
        id_vars=id_vars,
        value_vars=questions,
    )
    comms = ai_labels.melt(
        id_vars=id_vars,
        value_vars=comment_cols,
        value_name="comment",
    ).assign(variable=lambda df: df["variable"].str.replace("_comment", ""))
    long = pd.merge(ans, comms, on=id_vars + ["variable"])
    return long


def model_label_fpath(folder: str) -> Path:
    fpath = (
        Path(os.environ["DSS_HOME"])
        / "toxicainment"
        / "simple_inference_benchmark_results"
        / folder
        / "model_labels.csv"
    )
    return fpath


def difficulty_score(df: pd.DataFrame):
    difficulty = (
        (1 - ((df - 0.5).abs() / 0.5)).mean(axis=1).sort_values(ascending=False)
    )
    return difficulty


def krippendorf_alpha(post_ids: Iterable[str], responses: Iterable[Any]):
    value_counts = pd.crosstab(post_ids, responses).to_numpy()
    alpha = krippendorff.alpha(
        value_counts=value_counts,
        level_of_measurement="nominal",
    )
    return alpha


def plot_scalars_for_questions(
    scalars: list[float], questions: list[str], title: str, x_reversed: bool = False
):
    if x_reversed:
        questions = list(reversed(questions))
        scalars = list(reversed(scalars))
    fig = plt.figure(figsize=(4, 4))
    plt.bar(questions, scalars)
    plt.xticks(rotation=30, ha="right")
    plt.ylabel(title)
    plt.ylim(0, max(1, max(scalars) * 1.1))
    plt.tight_layout()
    plt.grid()
    plt.close()  # Close the figure to prevent double display
    return fig


def fleiss_kappa(table, method="fleiss"):
    """Copied from the statsmodels library to return p_rat as well"""
    """Fleiss' and Randolph's kappa multi-rater agreement measure

    Parameters
    ----------
    table : array_like, 2-D
        assumes subjects in rows, and categories in columns. Convert raw data
        into this format by using
        :func:`statsmodels.stats.inter_rater.aggregate_raters`
    method : str
        Method 'fleiss' returns Fleiss' kappa which uses the sample margin
        to define the chance outcome.
        Method 'randolph' or 'uniform' (only first 4 letters are needed)
        returns Randolph's (2005) multirater kappa which assumes a uniform
        distribution of the categories to define the chance outcome.

    Returns
    -------
    kappa : float
        Fleiss's or Randolph's kappa statistic for inter rater agreement

    Notes
    -----
    no variance or hypothesis tests yet

    Interrater agreement measures like Fleiss's kappa measure agreement relative
    to chance agreement. Different authors have proposed ways of defining
    these chance agreements. Fleiss' is based on the marginal sample distribution
    of categories, while Randolph uses a uniform distribution of categories as
    benchmark. Warrens (2010) showed that Randolph's kappa is always larger or
    equal to Fleiss' kappa. Under some commonly observed condition, Fleiss' and
    Randolph's kappa provide lower and upper bounds for two similar kappa_like
    measures by Light (1971) and Hubert (1977).

    References
    ----------
    Wikipedia https://en.wikipedia.org/wiki/Fleiss%27_kappa

    Fleiss, Joseph L. 1971. "Measuring Nominal Scale Agreement among Many
    Raters." Psychological Bulletin 76 (5): 378-82.
    https://doi.org/10.1037/h0031619.

    Randolph, Justus J. 2005 "Free-Marginal Multirater Kappa (multirater
    K [free]): An Alternative to Fleiss' Fixed-Marginal Multirater Kappa."
    Presented at the Joensuu Learning and Instruction Symposium, vol. 2005
    https://eric.ed.gov/?id=ED490661

    Warrens, Matthijs J. 2010. "Inequalities between Multi-Rater Kappas."
    Advances in Data Analysis and Classification 4 (4): 271-86.
    https://doi.org/10.1007/s11634-010-0073-4.
    """

    table = 1.0 * np.asarray(table)  # avoid integer division
    n_sub, n_cat = table.shape
    n_total = table.sum()
    n_rater = table.sum(1)
    n_rat = n_rater.max()
    # assume fully ranked
    assert n_total == n_sub * n_rat

    # marginal frequency  of categories
    p_cat = table.sum(0) / n_total

    table2 = table * table
    p_rat = (table2.sum(1) - n_rat) / (n_rat * (n_rat - 1.0))
    p_mean = p_rat.mean()

    if method == "fleiss":
        p_mean_exp = (p_cat * p_cat).sum()
    elif method.startswith("rand") or method.startswith("unif"):
        p_mean_exp = 1 / n_cat

    kappa = (p_mean - p_mean_exp) / (1 - p_mean_exp)
    return kappa, p_rat


def calc_bertscore(preds: list[str], refs: list[str]) -> float:
    P, R, F1 = score(preds, refs, lang="en", verbose=False, device="cuda")
    return np.mean(F1.tolist())


def bertscore_alignment(
    df: pd.DataFrame,
    reference_model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct"
) -> pd.DataFrame:
    
    COMMENT_CATEGORIES = [
        "is_eudaimonic_entertainment_comment",
        "is_hedonic_entertainment_comment",
        "is_intolerant_comment",
        "is_political_comment",
        "is_saxony_comment",
    ]

    per_model_scores = []

    for post_id, group in df.groupby("post_id"):
        ref_row = group[group["Model ID"] == reference_model_id]
        if ref_row.empty:
            continue

        for model_id in group["Model ID"].unique():
            if model_id == reference_model_id:
                continue

            model_row = group[group["Model ID"] == model_id]
            if model_row.empty:
                continue

            total_score = 0
            valid_categories = 0

            for category in COMMENT_CATEGORIES:
                ref_comment = ref_row[category].values[0]
                model_comment = model_row[category].values[0]

                if not isinstance(ref_comment, str) or not isinstance(model_comment, str):
                    continue
                if not ref_comment.strip() or not model_comment.strip():
                    continue

                score_val = calc_bertscore([model_comment], [ref_comment])
                if not np.isnan(score_val):
                    total_score += score_val
                    valid_categories += 1

            if valid_categories:
                per_model_scores.append({
                    "post_id": post_id,
                    "compared_model_id": model_id,
                    "bertscore_alignment": total_score / valid_categories
                })

    model_score_df = pd.DataFrame(per_model_scores)

    overall_alignment_df = (
        model_score_df
        .groupby("compared_model_id")["bertscore_alignment"]
        .mean()
        .reset_index()
        .rename(columns={"bertscore_alignment": "overall_alignment"})
    )

    return overall_alignment_df
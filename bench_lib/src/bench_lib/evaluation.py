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
from matplotlib import gridspec

logger = logging.getLogger(__name__)


def plot_runtime_side_by_side(
    rtmetrics_model1: pd.DataFrame,
    rtmetrics_model2: pd.DataFrame,
    title1: str = "",
    title2: str = "",
) -> plt.Figure:
    fig = plt.figure(figsize=(8, 3))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1, 0.05], wspace=0.3)

    xmax = max(
        rtmetrics_model1["Tokens_Generated"].max(),
        rtmetrics_model2["Tokens_Generated"].max(),
    )
    ymax = max(
        rtmetrics_model1["Model_Runtime"].max(),
        rtmetrics_model2["Model_Runtime"].max(),
    )
    ax1 = fig.add_subplot(gs[0], xlim=(0, xmax), ylim=(0, ymax))
    ax2 = fig.add_subplot(gs[1], xlim=(0, xmax), ylim=(0, ymax))
    visualize_runtime(
        rtmetrics_model1,
        figax=(fig, ax1),
        plot_colorbar=False,
        title=title1,
    )
    _, scatter = visualize_runtime(
        rtmetrics_model2,
        figax=(fig, ax2),
        plot_ylabel=False,
        plot_colorbar=False,
        title=title2,
    )

    scatter_cb = ax1.scatter([], [], c=[], cmap=scatter.cmap, norm=scatter.norm)
    cbar = fig.colorbar(scatter_cb, cax=fig.add_subplot(gs[2]))
    cbar.set_label("Input Frames Per Video")
    return fig


def visualize_runtime(
    df: pd.DataFrame,
    figax=None,
    plot_colorbar: bool = True,
    plot_ylabel: bool = True,
    title: str = "",
) -> tuple[plt.Figure, plt.Axes]:
    # The model takes longer for more input frames and for more output tokens generated
    if figax is None:
        fig = plt.figure(figsize=(6, 4))
        ax = fig.add_subplot(111)
    else:
        fig, ax = figax

    # Create scatter plot with alpha
    scatter = ax.scatter(
        df["Tokens_Generated"], df["Model_Runtime"], c=df["Total_Frames"], alpha=0.5
    )

    if plot_colorbar:
        # Create a separate scatter plot so that the colorbar is not affected by the alpha
        scatter_cb = ax.scatter([], [], c=[], cmap=scatter.cmap, norm=scatter.norm)
        _ = fig.colorbar(scatter_cb, ax=ax, label="Input Frames Per Video")

    if title != "":
        ax.set_title(title)

    ax.set_xlim(0, None)
    ax.set_ylim(0, None)
    ax.set_xlabel("Output Tokens Generated")
    if plot_ylabel:
        ax.set_ylabel("Model Runtime")
    ax.grid(alpha=0.5)
    fig.tight_layout()
    plt.close()  # Close the figure to prevent double display
    return fig, scatter


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
    fpath = benchmark_results_folder() / folder / "model_labels.csv"
    return fpath


def benchmark_results_folder() -> Path:
    return (
        Path(os.environ["DSS_HOME"])
        / "toxicainment"
        / "simple_inference_benchmark_results"
    )


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
    df: pd.DataFrame, reference_model_id: str = "Qwen/Qwen2.5-VL-72B-Instruct"
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

                if not isinstance(ref_comment, str) or not isinstance(
                    model_comment, str
                ):
                    continue
                if not ref_comment.strip() or not model_comment.strip():
                    continue

                score_val = calc_bertscore([model_comment], [ref_comment])
                if not np.isnan(score_val):
                    total_score += score_val
                    valid_categories += 1

            if valid_categories:
                per_model_scores.append(
                    {
                        "post_id": post_id,
                        "compared_model_id": model_id,
                        "bertscore_alignment": total_score / valid_categories,
                    }
                )

    model_score_df = pd.DataFrame(per_model_scores)

    overall_alignment_df = (
        model_score_df.groupby("compared_model_id")["bertscore_alignment"]
        .mean()
        .reset_index()
        .rename(columns={"bertscore_alignment": "overall_alignment"})
    )

    return overall_alignment_df


def plot_alignment_table(
    df: pd.DataFrame, reference_label: str = "Qwen 72B (reference)"
) -> plt.Figure:
    df_sorted = df.sort_values("overall_alignment", ascending=False)
    table_data = [
        [model, f"{score:.4f}"]
        for model, score in zip(
            df_sorted["compared_model_id"], df_sorted["overall_alignment"]
        )
    ]

    col_labels = ["BERTScore", reference_label]

    fig, ax = plt.subplots(figsize=(6, 1 + 0.4 * len(table_data)))
    ax.axis("off")

    table = ax.table(
        cellText=table_data,
        colLabels=col_labels,
        cellLoc="center",
        loc="center",
        colWidths=[0.5, 0.5],
    )

    table.auto_set_font_size(False)
    table.set_fontsize(12)
    table.scale(1, 1.5)

    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_fontsize(13)
            cell.set_text_props(weight="bold")
            cell.set_facecolor("#f0f0f0")

    plt.tight_layout()
    plt.close()
    return fig


def count_label_flips_per_post(
    merged_df: pd.DataFrame, label: str, n_runs: int
) -> pd.Series:
    label_cols = [f"{label}_run{i}" for i in range(n_runs)]
    existing_cols = [col for col in label_cols if col in merged_df.columns]
    label_data = merged_df[existing_cols]
    diffs = label_data.diff(axis=1).abs()  # column-wise absolute differences among runs
    return diffs.apply(lambda row: row.sum(), axis=1)


def check_self_consistency(
    csv_paths: list[str],
    model_to_check: str,
    n_runs: int,
) -> pd.DataFrame:
    LABEL_CATEGORIES = [
        "is_eudaimonic_entertainment",
        "is_hedonic_entertainment",
        "is_intolerant",
        "is_political",
        "is_saxony",
    ]

    model_labels_dfs = []

    # we add another run_id for number of experiments
    for run_id, path in enumerate(csv_paths):
        df = pd.read_csv(path)
        model_df = df[df["Model ID"] == model_to_check].copy()
        model_df["run_id"] = run_id
        model_df = model_df.dropna(subset=LABEL_CATEGORIES)
        for category in LABEL_CATEGORIES:
            model_df = model_df[model_df[category].notna()]
            model_df[category] = (
                model_df[category]
                .replace({"no": 0, "No": 0, "0": 0, "yes": 1, "Yes": 1, "1": 1})
                .astype(int)
            )
        if not model_df.empty:
            model_labels_dfs.append(model_df[["post_id", "run_id"] + LABEL_CATEGORIES])

    wide_dfs = []
    for run_id, df in enumerate(model_labels_dfs):
        wide_df = df.set_index("post_id")[LABEL_CATEGORIES].add_suffix(f"_run{run_id}")
        wide_dfs.append(wide_df)

    merged = pd.concat(wide_dfs, axis=1, join="outer")

    flip_counts_all = pd.DataFrame(index=merged.index)

    # calculate flip counts per post for the given model
    for label in LABEL_CATEGORIES:
        flip_counts_all[f"{label}_flip_count"] = count_label_flips_per_post(
            merged, label, n_runs
        )

    flip_counts_all = flip_counts_all.reset_index()
    flip_count_cols = [f"{label}_flip_count" for label in LABEL_CATEGORIES]
    total_flips = flip_counts_all[flip_count_cols].sum().sum()
    max_possible_flips = (n_runs - 1) * len(LABEL_CATEGORIES) * len(flip_counts_all)
    consistency_score = 1 - (total_flips / max_possible_flips)

    return flip_counts_all, consistency_score

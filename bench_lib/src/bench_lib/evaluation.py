from typing import Any, Iterable
import krippendorff
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from bench_lib.utils import Cols


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
    folders: list[str], questions: list[str], comment_cols: list[str]
) -> pd.DataFrame:
    all_ai_labels = []
    for folder in folders:
        ai_labels = pd.read_csv(
            f"../results/{folder}/model_labels.csv", dtype={"post_id": str}
        )
        ai_labels = ai_labels[
            [Cols.run_id, Cols.model_id, Cols.post_id, *questions, *comment_cols]
        ]
        rows_w_na = ai_labels[questions].isna().any(axis=1)
        print(f"{folder}: filtering {rows_w_na.sum()} rows with NA values")
        complete_ai_labels = ai_labels[~rows_w_na]
        print(f"{folder}: {len(complete_ai_labels)} rows after filtering")
        all_ai_labels.append(complete_ai_labels.assign(model_id=folder))

    all_ai_labels = pd.concat(all_ai_labels)
    return all_ai_labels


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


def plot_scalars_for_questions(scalars: list[float], questions: list[str], title: str):
    fig = plt.figure(figsize=(4, 4))
    plt.bar(questions, scalars)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(title)
    plt.ylim(0, 1)
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

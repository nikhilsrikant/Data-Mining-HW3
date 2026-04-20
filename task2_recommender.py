#!/usr/bin/env python3
"""Movie recommender assignment using Surprise.

Covers:
(a) reading ratings_small.csv
(b) evaluating with MAE and RMSE
(c) 5-fold CV for PMF, user-based CF, item-based CF
(d) comparing mean performance
(e) similarity sweep: cosine / MSD / Pearson
(f) neighbor sweep for user-based and item-based CF
(g) identifying best K by RMSE

Usage:
    python recommender_assignment_surprise.py \
        --ratings /path/to/ratings_small.csv \
        --outdir results

Notes:
- Uses Surprise's KNNBasic for user/item collaborative filtering.
- Uses Surprise's SVD with biased=False as PMF-equivalent.
- The timestamp column is read from the CSV, but not used by these algorithms.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from surprise import Dataset, KNNBasic, Reader, SVD, accuracy
from surprise.model_selection import KFold

RANDOM_STATE = 42
N_SPLITS = 5
DEFAULT_K = 40
SIMILARITIES = ["cosine", "msd", "pearson"]
K_VALUES = list(range(5, 101, 5))


def load_ratings(path: Path) -> Tuple[pd.DataFrame, Dataset]:
    """Load ratings_small.csv and convert it into a Surprise dataset."""
    df = pd.read_csv(path)

    required_cols = ["userId", "movieId", "rating"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Found columns: {list(df.columns)}")

    # Keep only what Surprise needs. Casting ids to string avoids any ambiguity
    # around raw-id handling across libraries and outputs.
    ratings = df[["userId", "movieId", "rating"]].copy()
    ratings["userId"] = ratings["userId"].astype(str)
    ratings["movieId"] = ratings["movieId"].astype(str)

    # Use the observed rating bounds from the file.
    rating_min = float(ratings["rating"].min())
    rating_max = float(ratings["rating"].max())
    reader = Reader(rating_scale=(rating_min, rating_max))
    data = Dataset.load_from_df(ratings[["userId", "movieId", "rating"]], reader)
    return ratings, data


def make_cv() -> KFold:
    """Return a reproducible 5-fold splitter."""
    return KFold(n_splits=N_SPLITS, random_state=RANDOM_STATE, shuffle=True)


def evaluate_algo(algo, data: Dataset) -> Dict[str, object]:
    """Evaluate one Surprise algorithm with reproducible 5-fold CV."""
    rmses: List[float] = []
    maes: List[float] = []

    for trainset, testset in make_cv().split(data):
        algo.fit(trainset)
        predictions = algo.test(testset)
        rmses.append(float(accuracy.rmse(predictions, verbose=False)))
        maes.append(float(accuracy.mae(predictions, verbose=False)))

    return {
        "rmse_folds": rmses,
        "mae_folds": maes,
        "rmse_mean": float(np.mean(rmses)),
        "mae_mean": float(np.mean(maes)),
        "rmse_std": float(np.std(rmses, ddof=1)) if len(rmses) > 1 else 0.0,
        "mae_std": float(np.std(maes, ddof=1)) if len(maes) > 1 else 0.0,
    }


def evaluate_main_models(data: Dataset) -> pd.DataFrame:
    """Part (c) and (d): PMF, user CF, item CF."""
    models = {
        "PMF": SVD(biased=False, random_state=RANDOM_STATE),
        "UserCF_MSD_k40": KNNBasic(
            k=DEFAULT_K,
            sim_options={"name": "msd", "user_based": True},
            verbose=False,
        ),
        "ItemCF_MSD_k40": KNNBasic(
            k=DEFAULT_K,
            sim_options={"name": "msd", "user_based": False},
            verbose=False,
        ),
    }

    rows = []
    for model_name, algo in models.items():
        result = evaluate_algo(algo, data)
        rows.append(
            {
                "model": model_name,
                "mean_RMSE": result["rmse_mean"],
                "std_RMSE": result["rmse_std"],
                "mean_MAE": result["mae_mean"],
                "std_MAE": result["mae_std"],
                **{f"RMSE_fold_{i+1}": v for i, v in enumerate(result["rmse_folds"])},
                **{f"MAE_fold_{i+1}": v for i, v in enumerate(result["mae_folds"])},
            }
        )

    return pd.DataFrame(rows).sort_values("mean_RMSE", ignore_index=True)


def evaluate_similarity_sweep(data: Dataset) -> pd.DataFrame:
    """Part (e): compare cosine, MSD, Pearson for user/item CF.

    We keep k fixed at DEFAULT_K so the only changing factor is the similarity.
    """
    rows = []
    for mode_name, user_based in [("UserCF", True), ("ItemCF", False)]:
        for sim_name in SIMILARITIES:
            algo = KNNBasic(
                k=DEFAULT_K,
                sim_options={"name": sim_name, "user_based": user_based},
                verbose=False,
            )
            result = evaluate_algo(algo, data)
            rows.append(
                {
                    "mode": mode_name,
                    "similarity": sim_name,
                    "k": DEFAULT_K,
                    "mean_RMSE": result["rmse_mean"],
                    "mean_MAE": result["mae_mean"],
                    "std_RMSE": result["rmse_std"],
                    "std_MAE": result["mae_std"],
                }
            )
    return pd.DataFrame(rows)


def evaluate_k_sweep(data: Dataset, similarity: str = "msd") -> pd.DataFrame:
    """Part (f) and (g): vary k while keeping similarity fixed.

    Using a fixed similarity isolates the effect of the number of neighbors.
    """
    rows = []
    for mode_name, user_based in [("UserCF", True), ("ItemCF", False)]:
        for k in K_VALUES:
            algo = KNNBasic(
                k=k,
                sim_options={"name": similarity, "user_based": user_based},
                verbose=False,
            )
            result = evaluate_algo(algo, data)
            rows.append(
                {
                    "mode": mode_name,
                    "similarity": similarity,
                    "k": k,
                    "mean_RMSE": result["rmse_mean"],
                    "mean_MAE": result["mae_mean"],
                    "std_RMSE": result["rmse_std"],
                    "std_MAE": result["mae_std"],
                }
            )
    return pd.DataFrame(rows)


def save_line_plot(
    df: pd.DataFrame,
    x_col: str,
    y_cols: Iterable[str],
    title: str,
    xlabel: str,
    ylabel: str,
    out_path: Path,
    x_labels: List[str] | None = None,
) -> None:
    """Save a simple line plot."""
    plt.figure(figsize=(8, 5))
    x_values = list(df[x_col])

    for y_col in y_cols:
        plt.plot(x_values, df[y_col], marker="o", label=y_col)

    if x_labels is not None:
        plt.xticks(x_values, x_labels)

    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_similarity_plots(sim_df: pd.DataFrame, outdir: Path) -> None:
    """Create one plot per CF mode plus one combined comparison plot (Q2e)."""
    order = {name: i for i, name in enumerate(SIMILARITIES)}
    working = sim_df.copy()
    working["sim_order"] = working["similarity"].map(order)
    working = working.sort_values(["mode", "sim_order"], ignore_index=True)

    # Individual plots (one per mode)
    for mode in ["UserCF", "ItemCF"]:
        subset = working[working["mode"] == mode].copy()
        subset["x"] = np.arange(len(subset))
        save_line_plot(
            df=subset,
            x_col="x",
            y_cols=["mean_RMSE", "mean_MAE"],
            title=f"{mode}: effect of similarity (k={DEFAULT_K})",
            xlabel="Similarity",
            ylabel="Error",
            out_path=outdir / f"similarity_{mode.lower()}.png",
            x_labels=subset["similarity"].tolist(),
        )

    # Combined RMSE comparison plot — answers Q2e "is the impact consistent?"
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_mode = {"UserCF": "#2196F3", "ItemCF": "#FF5722"}
    for ax, metric in zip(axes, ["mean_RMSE", "mean_MAE"]):
        for mode in ["UserCF", "ItemCF"]:
            subset = working[working["mode"] == mode].sort_values("sim_order")
            ax.plot(
                subset["similarity"],
                subset[metric],
                marker="o",
                label=mode,
                color=colors_mode[mode],
                linewidth=2,
            )
        ax.set_title(f"Q2e: {metric.replace('mean_', '')} by Similarity Metric", fontweight="bold")
        ax.set_xlabel("Similarity Metric")
        ax.set_ylabel(metric.replace("mean_", ""))
        ax.legend()
    plt.suptitle(f"User-CF vs Item-CF: Similarity Impact Comparison (k={DEFAULT_K})", fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "similarity_combined.png", dpi=200)
    plt.close()


def save_k_plots(k_df: pd.DataFrame, outdir: Path) -> None:
    """Create one plot per CF mode plus one combined comparison plot (Q2f/g)."""
    # Individual plots
    for mode in ["UserCF", "ItemCF"]:
        subset = k_df[k_df["mode"] == mode].sort_values("k", ignore_index=True)
        save_line_plot(
            df=subset,
            x_col="k",
            y_cols=["mean_RMSE", "mean_MAE"],
            title=f"{mode}: effect of number of neighbors",
            xlabel="Number of neighbors (k)",
            ylabel="Error",
            out_path=outdir / f"neighbors_{mode.lower()}.png",
        )

    # Combined plot — directly answers Q2f/g comparison question
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    colors_mode = {"UserCF": "#2196F3", "ItemCF": "#FF5722"}
    for ax, metric in zip(axes, ["mean_RMSE", "mean_MAE"]):
        for mode in ["UserCF", "ItemCF"]:
            subset = k_df[k_df["mode"] == mode].sort_values("k")
            best_k = subset.loc[subset[metric].idxmin(), "k"]
            best_v = subset[metric].min()
            ax.plot(subset["k"], subset[metric], marker="o", label=mode,
                    color=colors_mode[mode], linewidth=2)
            ax.axvline(x=best_k, linestyle="--", color=colors_mode[mode], alpha=0.5)
            ax.annotate(f"K={best_k}", xy=(best_k, best_v),
                        xytext=(best_k + 1, best_v + 0.005),
                        fontsize=8, color=colors_mode[mode])
        ax.set_title(f"Q2f/g: {metric.replace('mean_', '')} vs K", fontweight="bold")
        ax.set_xlabel("Number of Neighbors (K)")
        ax.set_ylabel(metric.replace("mean_", ""))
        ax.legend()
    plt.suptitle("User-CF vs Item-CF: Neighbor Count Impact", fontweight="bold")
    plt.tight_layout()
    plt.savefig(outdir / "neighbors_combined.png", dpi=200)
    plt.close()


def save_model_comparison_plot(main_df: pd.DataFrame, outdir: Path) -> None:
    """Bar chart comparing PMF, UserCF, ItemCF on RMSE and MAE (Q2d)."""
    models = main_df["model"].tolist()
    x = np.arange(len(models))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9, 5))
    bars1 = ax.bar(x - width / 2, main_df["mean_RMSE"], width, label="RMSE",
                   color="#2196F3", edgecolor="black")
    bars2 = ax.bar(x + width / 2, main_df["mean_MAE"],  width, label="MAE",
                   color="#FF5722", edgecolor="black")
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", fontsize=8)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{bar.get_height():.4f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=9)
    ax.set_title("Q2d: Average RMSE and MAE by Model (5-Fold CV)", fontweight="bold")
    ax.set_ylabel("Error")
    ax.legend()
    plt.tight_layout()
    plt.savefig(outdir / "model_comparison.png", dpi=200)
    plt.close()


def best_rows_by_metric(df: pd.DataFrame, group_col: str, metric_col: str) -> pd.DataFrame:
    """Return the best row per group according to a metric (lower is better)."""
    idx = df.groupby(group_col)[metric_col].idxmin()
    return df.loc[idx].sort_values(group_col).reset_index(drop=True)


def write_text_summary(
    ratings: pd.DataFrame,
    main_df: pd.DataFrame,
    sim_df: pd.DataFrame,
    k_df: pd.DataFrame,
    outdir: Path,
) -> None:
    """Save a concise text summary that answers the assignment questions."""
    best_main_rmse = main_df.loc[main_df["mean_RMSE"].idxmin()]
    best_main_mae = main_df.loc[main_df["mean_MAE"].idxmin()]

    best_sim = best_rows_by_metric(sim_df, "mode", "mean_RMSE")
    best_k = best_rows_by_metric(k_df, "mode", "mean_RMSE")

    lines = []
    lines.append("Movie recommender assignment summary")
    lines.append("=" * 40)
    lines.append("")
    lines.append(f"Number of ratings: {len(ratings):,}")
    lines.append(f"Number of users: {ratings['userId'].nunique():,}")
    lines.append(f"Number of movies: {ratings['movieId'].nunique():,}")
    lines.append(f"Observed rating range: {ratings['rating'].min()} to {ratings['rating'].max()}")
    lines.append("")
    lines.append("(c) 5-fold CV for PMF, UserCF, ItemCF")
    lines.append(main_df.to_string(index=False))
    lines.append("")
    lines.append("(d) Best main model")
    lines.append(
        f"Best by mean RMSE: {best_main_rmse['model']} (RMSE={best_main_rmse['mean_RMSE']:.4f}, MAE={best_main_rmse['mean_MAE']:.4f})"
    )
    lines.append(
        f"Best by mean MAE: {best_main_mae['model']} (RMSE={best_main_mae['mean_RMSE']:.4f}, MAE={best_main_mae['mean_MAE']:.4f})"
    )
    lines.append("")
    lines.append("(e) Best similarity by RMSE")
    lines.append(best_sim.to_string(index=False))
    lines.append("")
    lines.append("(f) and (g) Best k by RMSE (with fixed similarity='msd')")
    lines.append(best_k.to_string(index=False))
    lines.append("")
    same_k = len(best_k["k"].unique()) == 1
    lines.append(f"Is the best k the same for UserCF and ItemCF? {'Yes' if same_k else 'No'}")

    (outdir / "summary.txt").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate recommender models on ratings_small.csv")
    parser.add_argument("--ratings", type=Path, required=True, help="Path to ratings_small.csv")
    parser.add_argument("--outdir", type=Path, default=Path("results"), help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.outdir.mkdir(parents=True, exist_ok=True)

    ratings, data = load_ratings(args.ratings)

    print("Loaded ratings file")
    print(f"Ratings: {len(ratings):,}")
    print(f"Users:   {ratings['userId'].nunique():,}")
    print(f"Movies:  {ratings['movieId'].nunique():,}")
    print(f"Rating range: {ratings['rating'].min()} to {ratings['rating'].max()}")
    print()

    print("Evaluating main models...")
    main_df = evaluate_main_models(data)
    main_df.to_csv(args.outdir / "main_model_comparison.csv", index=False)
    save_model_comparison_plot(main_df, args.outdir)
    print(main_df[["model", "mean_RMSE", "mean_MAE"]].to_string(index=False))
    print()

    print("Evaluating similarity sweep...")
    sim_df = evaluate_similarity_sweep(data)
    sim_df.to_csv(args.outdir / "similarity_results.csv", index=False)
    save_similarity_plots(sim_df, args.outdir)
    print(sim_df.sort_values(["mode", "mean_RMSE"]).to_string(index=False))
    print()

    print("Evaluating k sweep...")
    k_df = evaluate_k_sweep(data, similarity="msd")
    k_df.to_csv(args.outdir / "k_results.csv", index=False)
    save_k_plots(k_df, args.outdir)
    best_k_df = best_rows_by_metric(k_df, "mode", "mean_RMSE")
    print(best_k_df.to_string(index=False))
    print()

    write_text_summary(ratings, main_df, sim_df, k_df, args.outdir)
    print(f"Saved outputs to: {args.outdir.resolve()}")


if __name__ == "__main__":
    main()

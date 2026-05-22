import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.spatial import cKDTree

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


def compare_cluster_runs(
    gt_parquet_path: str, guessed_parquet_path: str, max_distance: float
):
    """
    Compares guessed cluster centroids against ground truth centroids using a distance threshold.

    Returns:
        tuple: (df_guessed_evaluated, df_gt_evaluated)
    """
    # Load datasets
    df_gt = pd.read_parquet(gt_parquet_path)
    df_guess = pd.read_parquet(guessed_parquet_path)

    coord_cols = ["centroid_x", "centroid_y", "centroid_z"]

    # Extract coordinate matrices
    gt_coords = df_gt[coord_cols].to_numpy()
    guess_coords = df_guess[coord_cols].to_numpy()

    # Build spatial index trees
    gt_tree = cKDTree(gt_coords)
    guess_tree = cKDTree(guess_coords)

    # 1. Evaluate Guesses: Find all GT indices within max_distance for each Guess
    # query_ball_point returns a list of indices for each row
    guess_matches = gt_tree.query_ball_point(guess_coords, r=max_distance)

    df_guess["match_status"] = [
        "Success" if len(matches) > 0 else "False Positive" for matches in guess_matches
    ]

    # 2. Evaluate Ground Truth: Find all Guess indices within max_distance for each GT
    gt_matches = guess_tree.query_ball_point(gt_coords, r=max_distance)

    df_gt["false_negative"] = [len(matches) == 0 for matches in gt_matches]

    log.info(
        f'\n{"=" * 58} Guess {"=" * 58}\n'
        f"{df_guess}\n\n"
        f'{"=" * 58} Ground truth {"=" * 58}\n'
        f"{df_gt}"
    )

    return df_guess, df_gt


correct = "../data/damage_metrics_2.parquet"
guess = "../data/damage_metrics.parquet"

compare_cluster_runs(correct, guess, 5)

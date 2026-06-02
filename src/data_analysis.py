import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import logging

from registration import Registration

from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class DataAnalysis:
    def __init__(self):
        self.reg = Registration()

    def compare_cluster_runs(
        self,
        gt_parquet_path: str,
        guessed_parquet_path: str,
        max_distance: float,
        compact_view=False,
    ):
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
            "Success" if len(matches) > 0 else "False Positive"
            for matches in guess_matches
        ]

        # 2. Evaluate Ground Truth: Find all Guess indices within max_distance for each GT
        gt_matches = guess_tree.query_ball_point(gt_coords, r=max_distance)

        df_gt["false_negative"] = [len(matches) == 0 for matches in gt_matches]

        match_len = len(df_guess["match_status"])
        total_matches = (df_guess["match_status"] == "Success").sum()
        total_fp = (df_guess["match_status"] == "False Positive").sum()
        total_fn = (df_gt["false_negative"] == "True").sum()

        log.info(
            f"Number of false positives: {total_fp}. Number of false negatives: {total_fn}"
        )

        if compact_view:
            if total_fn > 0 and total_fp > 0:
                log.info(
                    f'\n{"=" * 8} Guess {"=" * 8}\n'
                    f"{df_guess[['cluster_id', 'match_status']]}\n\n"
                    f'{"=" * 8} Ground truth {"=" * 8}\n'
                    f"{df_gt[['cluster_id', 'false_negative']]}"
                )
            elif total_fn == 0 and total_fp > 0:
                log.info(
                    f'\n{"=" * 8} Guess {"=" * 8}\n'
                    f"{df_guess[['cluster_id', 'match_status']]}"
                )
            elif total_fn > 0 and total_fp == 0:
                log.info(
                    f'{"=" * 8} Ground truth {"=" * 8}\n'
                    f"{df_gt[['cluster_id', 'false_negative']]}"
                )
            else:
                pass
        else:
            log.info(
                f'\n{"=" * 58} Guess {"=" * 58}\n'
                f"{df_guess}\n\n"
                f'{"=" * 58} Ground truth {"=" * 58}\n'
                f"{df_gt}"
            )

        return df_guess, df_gt

    # NOTE: This is basically testing the scaling, is this what we want to test?
    def run_velocity_monte_carlo(
        self, pcd, tgt, csv_path, num_iterations=100, noise_std=0.5
    ):
        """
        Runs a Monte Carlo simulation by adding Gaussian noise to the measured
        velocities and evaluating the resulting registration RMSE.
        """
        rmse_results = []
        fitness_results = []

        df_base = pd.read_csv(csv_path)

        for i in tqdm(range(num_iterations)):
            # 1. Deep copy baseline velocity to prevent noise accumulation across loops
            df_noisy = df_base.copy()

            # 2. Apply Gaussian noise to the speed column
            noise = np.random.normal(loc=0.0, scale=noise_std, size=len(df_noisy))
            df_noisy["Speed_mms"] += noise

            # 3. Execute velocity correction
            try:
                # Note: Ensure you fix the shape bug in PC_velocity_correction before calling
                pc_corrected = self.reg.velocity_correction(df_noisy, pcd)

                # 4. Calculate registration metrics
                ransac_result = self.reg.get_initial_guess(pc_corrected, tgt)
                fitness = ransac_result.fitness
                rmse = ransac_result.inlier_rmse

                rmse_results.append(rmse)
                fitness_results.append(fitness)

            except Exception as e:
                print(f"Simulation iteration {i} failed: {str(e)}")
                continue

        rmse_results = np.array(rmse_results)

        # 5. Output summary statistics
        print("\n================== Monte Carlo Simulation Results ==================")
        print(f"Iterations:        {len(rmse_results)}")
        print(f"Velocity Noise σ:  {noise_std} mm/s")
        print(f"Mean RMSE:         {np.mean(rmse_results):.4f}")
        print(f"Median RMSE:       {np.median(rmse_results):.4f}")
        print(f"Std Dev RMSE:      {np.std(rmse_results):.4f}")
        print(
            f"Min / Max RMSE:    {np.min(rmse_results):.4f} / {np.max(rmse_results):.4f}"
        )
        print("====================================================================")
        print(f"Mean fitness:         {np.mean(fitness_results):.4f}")
        print(f"Median fitness:       {np.median(fitness_results):.4f}")
        print(f"Std Dev fitness:      {np.std(fitness_results):.4f}")
        print(
            f"Min / Max fitness:    {np.min(fitness_results):.4f} / {np.max(fitness_results):.4f}"
        )
        print("====================================================================")

        return rmse_results, fitness_results

    def plot_monte_carlo_results(self, rmse_results, noise_std):
        """
        Plots a 2-panel diagnostic figure for Monte Carlo RMSE analysis.
        Left: Histogram with Mean, Median, and 95th percentile bounds.
        Right: Cumulative Distribution Function (CDF) for risk tolerance tracking.
        """
        if len(rmse_results) == 0:
            print("Error: No data to plot.")
            return

        # Calculate metrics
        mean_val = np.mean(rmse_results)
        median_val = np.median(rmse_results)
        p95_val = np.percentile(rmse_results, 95)

        # Configure plot style for clean engineering layout
        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        fig.suptitle(
            f"Monte Carlo Sensitvity Analysis ($\sigma_{{noise}}$ = {noise_std} mm/s)",
            fontsize=14,
            fontweight="bold",
        )

        # --- Panel 1: Frequency Distribution (Histogram) ---
        counts, bins, _ = ax1.hist(
            rmse_results,
            bins=20,
            edgecolor="black",
            alpha=0.7,
            color="#2b5c8f",
            density=False,
        )

        # Statistical indicators
        ax1.axvline(
            mean_val,
            color="crimson",
            linestyle="--",
            linewidth=2,
            label=f"Mean: {mean_val:.3f}",
        )
        ax1.axvline(
            median_val,
            color="orange",
            linestyle="-",
            linewidth=2,
            label=f"Median: {median_val:.3f}",
        )
        ax1.axvline(
            p95_val,
            color="black",
            linestyle=":",
            linewidth=2,
            label=f"95th Percentile: {p95_val:.3f}",
        )

        ax1.set_title("RMSE Probability Density Location")
        ax1.set_xlabel("Registration RMSE (mm)")
        ax1.set_ylabel("Frequency (Counts)")
        ax1.grid(True, linestyle=":", alpha=0.6)
        ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="none")

        # --- Panel 2: Cumulative Distribution Function (CDF) ---
        sorted_data = np.sort(rmse_results)
        y_values = np.arange(1, len(sorted_data) + 1) / len(sorted_data)

        ax2.plot(
            sorted_data,
            y_values,
            marker=".",
            linestyle="none",
            color="#2b5c8f",
            label="Empirical CDF",
        )

        # Draw 95% interception line
        ax2.axhline(0.95, color="black", linestyle=":", linewidth=1.5)
        ax2.axvline(p95_val, color="black", linestyle=":", linewidth=1.5)

        ax2.set_title("Cumulative Probability / Risk Profile")
        ax2.set_xlabel("Registration RMSE (mm)")
        ax2.set_ylabel("Probability ($P \leq X$)")
        ax2.set_ylim(-0.05, 1.05)
        ax2.grid(True, linestyle=":", alpha=0.6)

        # Annotate the 95% confidence limit
        ax2.text(
            p95_val,
            0.5,
            f" 95% of runs\n RMSE $\leq$ {p95_val:.3f}",
            verticalalignment="center",
            horizontalalignment="left",
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
        )

        plt.tight_layout()
        plt.show()

    def run_noise_grid_sweep(
        self,
        pcd,
        gt_pcd,
        csv_path,
        means_range,
        stds_range,
        mc_iterations=10,
        uniform_donwsample=False,
    ):
        """
        Sweeps a 2D grid of noise mean (bias) and std dev (jitter).
        Generates a heatmap of the resulting mean RMSE.
        """
        # Initialize grid matrix
        rmse_grid = np.zeros((len(stds_range), len(means_range)))
        fitness_grid = np.zeros((len(stds_range), len(means_range)))
        ransac_grid = np.full((len(stds_range), len(means_range), 4, 4), np.nan)
        pc_grid = np.empty((len(stds_range), len(means_range)), dtype=object)

        df_base = pd.read_csv(csv_path)

        if uniform_donwsample:
            tgt_reg = gt_pcd.uniform_down_sample(20)
            self.reg.set_voxel(tgt_reg, ratio=0.03)
        else:
            tgt_reg = self.reg.downsample(gt_pcd, ratio=0.001)
            self.reg.set_voxel(tgt_reg, ratio=0.03)

        # Iterate through the grid
        for s_idx, sigma in enumerate(tqdm(stds_range)):
            for m_idx, mean in enumerate(means_range):
                iter_rmse = []
                iter_fitness = []
                rsc_results = []
                pcds = []

                for _ in range(mc_iterations):
                    df_noisy = df_base.copy()

                    # Apply Gaussian noise with specific mean and std dev
                    noise = np.random.normal(loc=mean, scale=sigma, size=len(df_noisy))
                    df_noisy["Speed_mms"] += noise

                    try:
                        pc_corrected = self.reg.velocity_correction(df_noisy, pcd)
                        # pc_corrected = det.select_bus_hull(
                        #     pc_corrected, eps=2.0, visualise=False
                        # )

                        if uniform_donwsample:
                            pc_reg = pc_corrected.uniform_down_sample(20)
                        else:
                            pc_reg = self.reg.downsample(pc_corrected, ratio=0.001)

                        icp, _, _ = self.reg.register(pc_reg, tgt_reg, ransac_retries=3)

                        # 4. Calculate registration metrics
                        # icp, _, _ = reg.register(
                        #     pc_corrected, gt_pcd, ransac_retries=3
                        # )
                        # icp = reg.get_initial_guess(pc_corrected, gt_pcd)

                        eval = self.reg.evaluate_alignment(
                            pc_corrected, gt_pcd, icp.transformation
                        )

                        fitness = eval.fitness
                        rmse = eval.inlier_rmse

                        # fitness = icp.fitness
                        # rmse = icp.inlier_rmse

                        iter_fitness.append(fitness)
                        iter_rmse.append(rmse)

                        pc_corrected = pc_corrected.transform(icp.transformation)
                    except Exception:
                        continue

                # Store average performance for this specific noise profile
                rmse_grid[s_idx, m_idx] = np.mean(iter_rmse) if iter_rmse else np.nan
                fitness_grid[s_idx, m_idx] = (
                    np.mean(iter_fitness) if iter_fitness else np.nan
                )
                # ransac_grid[s_idx, m_idx, :, :] = r[0]
                pc_grid[s_idx, m_idx] = pc_corrected

        # Plotting the 2D Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(
            rmse_grid,
            xticklabels=[f"{m:.1f}" for m in means_range],
            yticklabels=[f"{s:.1f}" for s in stds_range],
            cmap="viridis",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Mean Registration RMSE (mm)"},
        )
        plt.title(
            "RMSE Sensitivity Matrix: Velocity Bias ($\mu$) vs. Jitter ($\sigma$)"
        )
        plt.xlabel("Velocity Noise Mean / Bias ($\mu$ in mm/s)")
        plt.ylabel("Velocity Noise Std Dev / Jitter ($\sigma$ in mm/s)")
        plt.gca().invert_yaxis()  # Low noise at bottom, high noise at top
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            fitness_grid,
            xticklabels=[f"{m:.1f}" for m in means_range],
            yticklabels=[f"{s:.1f}" for s in stds_range],
            cmap="viridis",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Mean Registration RMSE (mm)"},
        )
        plt.title(
            "Fitness Sensitivity Matrix: Velocity Bias ($\mu$) vs. Jitter ($\sigma$)"
        )
        plt.xlabel("Velocity Noise Mean / Bias ($\mu$ in mm/s)")
        plt.ylabel("Velocity Noise Std Dev / Jitter ($\sigma$ in mm/s)")
        plt.gca().invert_yaxis()  # Low noise at bottom, high noise at top
        plt.show()

        return rmse_grid, fitness_grid, pc_grid

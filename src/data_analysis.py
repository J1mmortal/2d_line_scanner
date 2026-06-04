import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import open3d as o3d
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import logging

from registration import Registration
from damage_detection import DamageDetector
from pipeline import Pipeline

from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator

plt.style.use("science")
# plt.style.use(["science", "ieee"])

log = logging.getLogger(__name__)

# plt.rcParams["text.latex.preamble"] = r"\usepackage{lmodern}"

# params = {
#     "text.usetex": True,
#     "font.size": 11,
#     "font.family": "lmodern",
# }
# plt.rcParams.update(params)


class DataAnalysis:
    def __init__(self, pipe_params):
        self.reg = Registration(5)
        self.det = DamageDetector()

        self.pipe_params = pipe_params

        self.sigma_thresh = 4
        self.percentile = 80

        self.cluster_eps = 1.4
        self.cluster_min_samples = 300

        self.gt_parquet_path = "../data/bus4_gt.parquet"
        self.guess_parquet_path = "../data/damage_metrics.parquet"

    def run_velocity_monte_carlo(
        self, src_path, tgt_path, csv_path, num_iterations=100, noise_std=0.5
    ):
        """
        Runs a Monte Carlo simulation by adding Gaussian noise to the measured
        velocities and evaluating the resulting registration RMSE.
        """
        rmse_results, fitness_results, fn_results, fp_results = [], [], [], []

        df_base = pd.read_csv(csv_path)

        for i in tqdm(range(num_iterations)):
            df_noisy = df_base.copy()

            # Apply Gaussian noise to the speed column
            noise = np.random.normal(loc=0.0, scale=noise_std, size=len(df_noisy))
            df_noisy["Speed_mms"] += noise

            try:
                pip = Pipeline(
                    source_path=src_path,
                    target_path=tgt_path,
                    speed_data=df_noisy,
                    **self.pipe_params,
                )

                metrics = pip.run()

                # FIX: Corrected inverted list insertions
                fitness_results.append(pip.fitness)
                rmse_results.append(pip.rmse)
                fn_results.append(pip.fn)
                fp_results.append(pip.fp)

            except Exception as e:
                print(f"Simulation iteration {i} failed: {str(e)}")
                continue

        rmse_results = np.array(rmse_results)
        fitness_results = np.array(fitness_results)
        fn_results = np.array(fn_results)
        fp_results = np.array(fp_results)

        print("\n================== Monte Carlo Simulation Results ==================")
        print(f"Iterations:        {len(rmse_results)}")
        print(f"Velocity Noise $\sigma$:  {noise_std} mm/s")
        print(
            f"Mean / Median RMSE: {np.mean(rmse_results):.4f} / {np.median(rmse_results):.4f} mm"
        )
        print(
            f"Mean / Median Fitness: {np.mean(fitness_results):.4f} / {np.median(fitness_results):.4f}"
        )
        print(
            f"Mean FP / FN Counts: {np.mean(fp_results):.2f} / {np.mean(fn_results):.2f}"
        )
        print("====================================================================")

        return rmse_results, fitness_results, fn_results, fp_results

    def run_time_sync_monte_carlo(
        self, src_path, tgt_path, csv_path, num_iterations=100, delay_std=0.01
    ):
        """
        Runs a Monte Carlo simulation by adding Gaussian time-synchronization
        delays (latency jitter) to the velocity profile timestamps and
        evaluating downstream geometric and classification metrics.

        Parameters:
        -----------
        delay_std : float
            The standard deviation of the time shift in seconds (e.g., 0.01 = 10ms).
        """
        rmse_results, fitness_results, fn_results, fp_results = [], [], [], []

        df_base = pd.read_csv(csv_path)

        for i in tqdm(range(num_iterations), desc="Time-Sync MC"):
            df_noisy = df_base.copy()

            # Generate a single constant time shift for this entire drive-by run
            tau = np.random.normal(loc=0.0, scale=delay_std)
            df_noisy["Time_s"] += tau

            try:
                pip = Pipeline(
                    source_path=src_path,
                    target_path=tgt_path,
                    speed_data=df_noisy,
                    **self.pipe_params,
                )

                pip.run()

                fitness_results.append(pip.fitness)
                rmse_results.append(pip.rmse)
                fn_results.append(pip.fn)
                fp_results.append(pip.fp)

            except Exception as e:
                log.warning(
                    f"Iteration {i} failed with delay {tau*1000:.2f}ms: {str(e)}"
                )
                continue

        rmse_results = np.array(rmse_results)
        fitness_results = np.array(fitness_results)
        fn_results = np.array(fn_results)
        fp_results = np.array(fp_results)

        print("\n================== Time-Sync Monte Carlo Results ==================")
        print(f"Iterations:              {len(rmse_results)}")
        print(f"Latency Jitter $\\sigma$:   {delay_std * 1000:.2f} ms")
        print(
            f"Mean / Median RMSE:      {np.mean(rmse_results):.4f} / {np.median(rmse_results):.4f} mm"
        )
        print(
            f"Mean / Median Fitness:   {np.mean(fitness_results):.4f} / {np.median(fitness_results):.4f}"
        )
        print(
            f"Mean FP / FN Counts:     {np.mean(fp_results):.2f} / {np.mean(fn_results):.2f}"
        )
        print("====================================================================")

        return rmse_results, fitness_results, fn_results, fp_results

    def plot_monte_carlo_results(
        self, rmse_res, fitness_res, fn_res, fp_res, noise_std
    ):
        """
        Plots a 4-panel diagnostic grid evaluating the probability density locations
        and risk parameters for all pipeline metrics under noise.
        """
        if len(rmse_res) == 0:
            print("Error: No data to plot.")
            return

        plt.rcParams["font.family"] = "sans-serif"
        plt.rcParams["xtick.direction"] = "in"
        plt.rcParams["ytick.direction"] = "in"

        # Initialize a balanced 2x2 subplot matrix
        fig, axs = plt.subplots(2, 2, figsize=(14, 10))
        fig.suptitle(
            f"Monte Carlo Pipeline Sensitivity Analysis ($\sigma_{{noise}}$ = {noise_std} mm/s)",
            fontsize=16,
            fontweight="bold",
        )

        # Dataset configuration mapping to cleanly loop through variables
        metrics_config = [
            {
                "data": rmse_res,
                "name": "Registration RMSE",
                "unit": "mm",
                "color": "#2b5c8f",
                "pos": (0, 0),
                "upper_risk": True,
            },
            {
                "data": fitness_res,
                "name": "ICP Alignment Fitness",
                "unit": "ratio",
                "color": "darkorange",
                "pos": (0, 1),
                "upper_risk": False,
            },
            {
                "data": fp_res,
                "name": "Segmentation False Positives (FP)",
                "unit": "counts",
                "color": "crimson",
                "pos": (1, 0),
                "upper_risk": True,
            },
            {
                "data": fn_res,
                "name": "Segmentation False Negatives (FN)",
                "unit": "counts",
                "color": "darkmagenta",
                "pos": (1, 1),
                "upper_risk": True,
            },
        ]

        for config in metrics_config:
            ax = axs[config["pos"]]
            data = config["data"]

            # Draw baseline probability density location histogram
            ax.hist(data, bins=20, edgecolor="black", alpha=0.7, color=config["color"])

            mean_val = np.mean(data)
            median_val = np.median(data)

            ax.axvline(
                mean_val,
                color="black",
                linestyle="--",
                linewidth=1.5,
                label=f"Mean: {mean_val:.2f}",
            )
            ax.axvline(
                median_val,
                color="blue",
                linestyle="-.",
                linewidth=1.5,
                label=f"Median: {median_val:.2f}",
            )

            # Determine risk tolerance bounds based on target metric direction
            if config["upper_risk"]:
                p95_val = np.percentile(data, 95)
                ax.axvline(
                    p95_val,
                    color="red",
                    linestyle=":",
                    linewidth=2,
                    label=f"95th Pctl: {p95_val:.2f}",
                )
            else:
                p05_val = np.percentile(data, 5)
                ax.axvline(
                    p05_val,
                    color="red",
                    linestyle=":",
                    linewidth=2,
                    label=f"5th Pctl: {p05_val:.2f}",
                )

            ax.set_title(f"{config['name']} Distribution")
            ax.set_xlabel(f"Value ({config['unit']})")
            ax.set_ylabel("Frequency (Counts)")
            ax.grid(True, linestyle=":", alpha=0.6)
            ax.legend(
                loc="upper right", frameon=True, facecolor="white", edgecolor="none"
            )

        plt.tight_layout()
        plt.savefig("monte_carlo_all_metrics_analysis.png", dpi=300)

    def run_noise_grid_sweep(
        self,
        src_path,
        tgt_path,
        csv_path,
        means_range,
        stds_range,
        mc_iterations=10,
        uniform_downsample=False,
        detect=True,
    ):
        """
        Sweeps a 2D grid of noise mean (bias) and std dev (jitter).
        Generates a heatmap of the resulting mean RMSE.
        """
        # Initialize grid matrix
        rmse_grid = np.zeros((len(stds_range), len(means_range)))
        fitness_grid = np.zeros((len(stds_range), len(means_range)))
        fp_grid = np.zeros((len(stds_range), len(means_range)))
        fn_grid = np.zeros((len(stds_range), len(means_range)))
        src_grid = np.empty((len(stds_range), len(means_range)), dtype=object)
        tgt_grid = np.empty((len(stds_range), len(means_range)), dtype=object)
        clst_grid = np.empty((len(stds_range), len(means_range)), dtype=object)

        df_base = pd.read_csv(csv_path)

        # Iterate through the grid
        for s_idx, sigma in enumerate(tqdm(stds_range, desc="Sigma sweep", position=0)):
            for m_idx, mean in enumerate(
                tqdm(means_range, desc="Mean sweep", position=1, leave=False)
            ):
                iter_rmse, iter_fitness, iter_fp, iter_fn = [], [], [], []

                for _ in range(mc_iterations):
                    df_noisy = df_base.copy()
                    log.info(
                        f"Adding noise with mean {mean} and standard deviation {sigma}"
                    )
                    noise = np.random.normal(loc=mean, scale=sigma, size=len(df_noisy))
                    df_noisy["Speed_mms"] += noise

                    try:
                        pip = Pipeline(
                            source_path=src_path,
                            target_path=tgt_path,
                            speed_data=df_noisy,
                            **self.pipe_params,
                        )

                        metrics = pip.run()

                        # Extract metrics populated inside the pipeline instance
                        iter_fitness.append(pip.fitness)
                        iter_rmse.append(pip.rmse)
                        iter_fp.append(pip.fp)
                        iter_fn.append(pip.fn)

                    except Exception as e:
                        log.warning(f"Iteration failed: {e}")
                        continue

                # Store average performance for this specific noise profile
                rmse_grid[s_idx, m_idx] = np.mean(iter_rmse) if iter_rmse else np.nan
                fitness_grid[s_idx, m_idx] = (
                    np.mean(iter_fitness) if iter_fitness else np.nan
                )
                fp_grid[s_idx, m_idx] = np.mean(iter_fp) if iter_fp else np.nan
                fn_grid[s_idx, m_idx] = np.mean(iter_fn) if iter_fn else np.nan

                # pip.tgt.paint_uniform_color([0.0, 0.0, 1.0])

                # src_grid[s_idx, m_idx] = self.reg.downsample(pip.alg_src, ratio=0.002)
                # tgt_grid[s_idx, m_idx] = self.reg.downsample(pip.tgt, ratio=0.002)

                cluster_cloud = self.det.color_point_cloud_by_labels(
                    pip.alg_src, pip.labels, downsample=0.001, write=False, vis=False
                )

                clst_grid[s_idx, m_idx] = cluster_cloud

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

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            fp_grid,
            xticklabels=[f"{m:.1f}" for m in means_range],
            yticklabels=[f"{s:.1f}" for s in stds_range],
            cmap="viridis",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Number of False positives"},
        )
        plt.title(
            "False Positive Sensitivity Matrix: Velocity Bias ($\mu$) vs. Jitter ($\sigma$)"
        )
        plt.xlabel("Velocity Noise Mean / Bias ($\mu$ in mm/s)")
        plt.ylabel("Velocity Noise Std Dev / Jitter ($\sigma$ in mm/s)")
        plt.gca().invert_yaxis()  # Low noise at bottom, high noise at top
        plt.show()

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            fn_grid,
            xticklabels=[f"{m:.1f}" for m in means_range],
            yticklabels=[f"{s:.1f}" for s in stds_range],
            cmap="viridis",
            annot=True,
            fmt=".3f",
            cbar_kws={"label": "Number of False negatives"},
        )
        plt.title(
            "False Negative Sensitivity Matrix: Velocity Bias ($\mu$) vs. Jitter ($\sigma$)"
        )
        plt.xlabel("Velocity Noise Mean / Bias ($\mu$ in mm/s)")
        plt.ylabel("Velocity Noise Std Dev / Jitter ($\sigma$ in mm/s)")
        plt.gca().invert_yaxis()  # Low noise at bottom, high noise at top
        plt.show()

        return rmse_grid, fitness_grid, src_grid, tgt_grid, fp_grid, fn_grid, clst_grid

    def run_noise_delay_sweep(
        self,
        src_path,
        tgt_path,
        csv_path,
        time_delays,
        mc_iterations=10,
    ):
        """
        Sweeps a 1D time delay on the velocity measurement
        """

        df_base = pd.read_csv(csv_path)

        rmse, fitness, fn, fp = [], [], [], []
        clusters = np.empty(len(time_delays), dtype=object)

        # Iterate through the grid
        for i, delay in enumerate(tqdm(time_delays, desc="Delay sweep", position=0)):

            iter_rmse, iter_fitness, iter_fp, iter_fn = [], [], [], []

            for _ in range(mc_iterations):

                # Load baseline dataframe
                df_delayed = df_base.copy()
                df_delayed["Time_s"] += delay

                log.info(f"Adding a time delay of {delay}")

                try:
                    pip = Pipeline(
                        source_path=src_path,
                        target_path=tgt_path,
                        speed_data=df_delayed,
                        **self.pipe_params,
                    )

                    metrics = pip.run()

                    # Extract metrics populated inside the pipeline instance
                    iter_fitness.append(pip.fitness)
                    iter_rmse.append(pip.rmse)
                    iter_fp.append(pip.fp)
                    iter_fn.append(pip.fn)

                except Exception as e:
                    log.warning(f"Iteration failed: {e}")
                    continue

            # Store average performance for this specific noise profile
            fitness.append(np.mean(iter_fitness) if iter_fitness else np.nan)
            rmse.append(np.mean(iter_rmse) if iter_rmse else np.nan)
            fn.append(np.mean(iter_fn) if iter_fn else np.nan)
            fp.append(np.mean(iter_fp) if iter_fp else np.nan)

            cluster_cloud = self.det.color_point_cloud_by_labels(
                pip.alg_src, pip.labels, downsample=0.001, write=False, vis=False
            )

            clusters[i] = cluster_cloud

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Geometric Alignment Metrics (Dual Y-Axes)
        color = "#2b5c8f"
        ax1.set_xlabel(r"Time Delay $\Delta t$ (s)")
        ax1.set_ylabel("Registration RMSE (mm)", color=color)
        line1 = ax1.plot(
            time_delays, rmse, color=color, marker="o", linewidth=2, label="Mean RMSE"
        )
        ax1.tick_params(axis="y", labelcolor=color)
        ax1.grid(True, linestyle=":", alpha=0.6)

        ax1_twin = ax1.twinx()
        color = "darkorange"
        ax1_twin.set_ylabel("ICP Alignment Fitness", color=color)
        line2 = ax1_twin.plot(
            time_delays,
            fitness,
            color=color,
            marker="s",
            linestyle="--",
            linewidth=2,
            label="Mean Fitness",
        )
        ax1_twin.tick_params(axis="y", labelcolor=color)

        # Unified legend for dual axis panel
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc=0)
        ax1.set_title("Registration Sensitivity to Temporal Delays")

        # Panel 2: Damage Classification Metrics (False Positives & False Negatives)
        ax2.plot(
            time_delays,
            fp,
            color="crimson",
            marker="v",
            linewidth=2,
            label="False Positives (FP)",
        )
        ax2.plot(
            time_delays,
            fn,
            color="black",
            marker="^",
            linewidth=2,
            linestyle=":",
            label="False Negatives (FN)",
        )
        ax2.set_xlabel(r"Time Delay $\Delta t$ (s)")
        ax2.set_ylabel("Defect Classification Error Count")
        ax2.set_title("Damage Segmentation Accuracy vs. Time Delay")
        ax2.grid(True, linestyle=":", alpha=0.6)
        ax2.legend(loc=0)

        plt.tight_layout()
        plt.savefig("time_delay_sweep_analysis.png", dpi=300)

        return rmse, fitness, fn, fp, clusters


# def detection_iteration(
#     self, df_noisy, pcd, gt_pcd, tgt_reg, uniform_downsample=True, detect=False
# ):
#     pc_corrected = self.reg.velocity_correction(
#         df_noisy, pcd, denoise=False, visualise=False
#     )
#     # pc_corrected = det.select_bus_hull(
#     #     pc_corrected, eps=2.0, visualise=False
#     # )

#     if uniform_downsample:
#         pc_reg = pc_corrected.uniform_down_sample(20)
#     else:
#         pc_reg = self.reg.downsample(pc_corrected, ratio=0.001)

#     icp, _, _ = self.reg.register(pc_reg, tgt_reg, ransac_retries=3)

#     eval = self.reg.evaluate_alignment(pc_corrected, gt_pcd, icp.transformation)

#     fitness = eval.fitness
#     rmse = eval.inlier_rmse

#     pc_corrected = pc_corrected.transform(icp.transformation)

#     if detect:
#         mask, distances, threshold = self.det.detect_damage(
#             pc_corrected,
#             gt_pcd,
#             sigma_thresh=self.sigma_thresh,
#             percentile=self.percentile,
#             bidirectional=True,  # new gate
#             outliers_points=200,
#             radius=2,
#         )

#         mask = self.det.crop_damage(pc_corrected, mask, 55, 30)

#         labels = self.det.cluster(
#             pc_corrected,
#             mask,
#             eps=self.cluster_eps,
#             min_samples=self.cluster_min_samples,
#             verbose=False,
#         )

#         metrics = self.det.calculate_damage_metrics(
#             pc_corrected, distances, labels, write_to_pd=True, verbose=False
#         )

#         n_clusters = len(set(labels[labels >= 0]))
#         log.info("Found %d damage cluster(s)", n_clusters)

#         number_fp, number_fn = self.det.compare_cluster_runs(
#             self.gt_parquet_path, self.guess_parquet_path, 5, True, True
#         )

#         return fitness, rmse, pc_corrected, number_fp, number_fn
#     else:
#         return fitness, rmse, pc_corrected

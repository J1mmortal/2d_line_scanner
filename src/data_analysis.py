import numpy as np
import matplotlib.pyplot as plt
import scienceplots
import open3d as o3d
import pandas as pd
import seaborn as sns
from tqdm import tqdm
import logging
import os

from scipy.spatial import cKDTree
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator
from matplotlib.ticker import MaxNLocator

from registration import Registration
from damage_detection import DamageDetector
from pipeline import Pipeline

plt.style.use("science")
# plt.style.use(["science", "ieee"])

log = logging.getLogger(__name__)


class DataAnalysis:
    def __init__(self, pipe_params):
        # Initialize core modules
        self.reg = Registration(5)
        self.det = DamageDetector()

        self.pipe_params = pipe_params

        # Define statistical thresholds for anomaly filtering
        self.sigma_thresh = 4
        self.percentile = 80

        # DBSCAN hyperparameters
        self.cluster_eps = 1.4
        self.cluster_min_samples = 300

        # File I/O targets
        self.gt_parquet_path = "../data/bus4_gt.parquet"
        self.guess_parquet_path = "../data/damage_metrics.parquet"

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
        output_dir="../data/sweep_results",
    ):
        """Executes a 2D parameter sweep over noise bias (mean) and variance (std)."""
        os.makedirs(output_dir, exist_ok=True)

        # 2D matrices for parameter grid combinations
        rmse_grid = np.zeros((len(stds_range), len(means_range)))
        fitness_grid = np.zeros((len(stds_range), len(means_range)))
        fp_grid = np.zeros((len(stds_range), len(means_range)))
        fn_grid = np.zeros((len(stds_range), len(means_range)))

        df_base = pd.read_csv(csv_path)

        # Map parameter combinations through nested loop structures
        for s_idx, sigma in enumerate(tqdm(stds_range, desc="Sigma sweep", position=0)):
            for m_idx, mean in enumerate(
                tqdm(means_range, desc="Mean sweep", position=1, leave=False)
            ):
                iter_rmse, iter_fitness, iter_fp, iter_fn = [], [], [], []

                for _ in range(mc_iterations):
                    df_noisy = df_base.copy()
                    noise = np.random.normal(loc=mean, scale=sigma, size=len(df_noisy))
                    log.info(
                        f"Adding noise with mean {mean} and standard deviation {sigma}"
                    )
                    df_noisy["Speed_mms"] += noise

                    try:
                        pip = Pipeline(
                            source_path=src_path,
                            target_path=tgt_path,
                            speed_data=df_noisy,
                            **self.pipe_params,
                        )
                        metrics = pip.run()

                        iter_fitness.append(pip.fitness)
                        iter_rmse.append(pip.rmse)
                        iter_fp.append(pip.fp)
                        iter_fn.append(pip.fn)

                    except Exception as e:
                        log.warning(f"Iteration failed: {e}")
                        continue

                # Aggregate results into matrices
                rmse_grid[s_idx, m_idx] = np.mean(iter_rmse) if iter_rmse else np.nan
                fitness_grid[s_idx, m_idx] = (
                    np.mean(iter_fitness) if iter_fitness else np.nan
                )
                fp_grid[s_idx, m_idx] = np.mean(iter_fp) if iter_fp else np.nan
                fn_grid[s_idx, m_idx] = np.mean(iter_fn) if iter_fn else np.nan

        # Write saved metrics to disk
        archive_path = os.path.join(output_dir, "sweep_metrics.npz")
        np.savez_compressed(
            archive_path,
            rmse_grid=rmse_grid,
            fitness_grid=fitness_grid,
            fp_grid=fp_grid,
            fn_grid=fn_grid,
            means_range=means_range,
            stds_range=stds_range,
        )
        print(f"Sweep numerical matrices saved successfully to {archive_path}")
        return archive_path

    def plot_saved_sweep_results(
        self, archive_path, figs_output_dir="../data/figs/latest"
    ):
        """Generates figures from saved sweep data."""
        os.makedirs(figs_output_dir, exist_ok=True)
        plt.style.use(["science", "ieee"])

        IEEE_COL_W = 3.5  # inches
        FONT_LABEL = 7  # axis label pt
        FONT_TICK = 6  # tick labels and annotations pt
        FONT_CBAR = 6  # colorbar label pt

        data = np.load(archive_path)
        means_range = data["means_range"]
        stds_range = data["stds_range"]
        n_cols = len(means_range)
        n_rows = len(stds_range)

        LABEL_MARGIN = 0.55  # inches for xlabel + colorbar top cap
        fig_h = IEEE_COL_W * (n_rows / n_cols) + LABEL_MARGIN

        # Define configurations for target heatmaps
        plots_config = [
            {
                "matrix": data["rmse_grid"],
                "label": r"Mean reg. RMSE (mm)",
                "fmt": ".3f",
                "cmap": "viridis",  # lower is better; dark = high error
                "file": "rmse_sensitivity_matrix_denoise.pdf",
            },
            {
                "matrix": data["fitness_grid"],
                "label": r"Mean fitness",
                "fmt": ".2f",
                "cmap": "viridis_r",  # reversed: dark = low fitness = bad
                "file": "fitness_sensitivity_matrix_denoise.pdf",
            },
            {
                "matrix": data["fp_grid"],
                "label": r"False positives",
                "fmt": ".1f",
                "cmap": "Reds",  # red scale: "more = worse" is intuitive
                "file": "fp_sensitivity_matrix_denoise.pdf",
            },
            {
                "matrix": data["fn_grid"],
                "label": r"False negatives",
                "fmt": ".1f",
                "cmap": "Reds",
                "file": "fn_sensitivity_matrix_denoise.pdf",
            },
        ]

        for cfg in plots_config:
            fig, ax = plt.subplots(figsize=(IEEE_COL_W, fig_h))

            sns.heatmap(
                cfg["matrix"],
                xticklabels=[f"{m:.2g}" for m in means_range],
                yticklabels=[f"{s:.2g}" for s in stds_range],
                cmap=cfg["cmap"],
                annot=True,
                fmt=cfg["fmt"],
                annot_kws={"size": FONT_TICK, "weight": "regular"},
                linewidths=0.4,  # cell borders for readability
                linecolor="white",
                square=True,
                cbar_kws={
                    "label": cfg["label"],
                    "shrink": 0.65,
                    "aspect": 20,  # thinner bar = less visual weight
                    "pad": 0.02,
                },
                ax=ax,
            )

            cbar = ax.collections[0].colorbar
            cbar.ax.tick_params(labelsize=FONT_TICK, length=2, pad=2)
            cbar.set_label(cfg["label"], size=FONT_CBAR, labelpad=4)

            ax.set_xlabel(
                r"Velocity noise bias $\mu$ (mm$\,$s$^{-1}$)",
                fontsize=FONT_LABEL,
                labelpad=3,
            )
            ax.set_ylabel(
                r"Velocity noise jitter $\sigma$ (mm$\,$s$^{-1}$)",
                fontsize=FONT_LABEL,
                labelpad=3,
            )
            ax.tick_params(
                axis="both", which="major", labelsize=FONT_TICK, length=2, pad=2
            )

            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.invert_yaxis()

            fig.savefig(
                os.path.join(figs_output_dir, cfg["file"]),
                bbox_inches="tight",
                dpi=300,
                transparent=True,
            )
            plt.close(fig)
            print(f"Generated: {cfg['file']}")

    def run_noise_delay_sweep(
        self,
        src_path,
        tgt_path,
        csv_path,
        time_delays,
        mc_iterations=10,
        output_dir="../data/sweep_results",
    ):
        """
        Sweeps a 1D time delay on the velocity input and saves the
        numerical arrays to a compressed archive.
        """
        os.makedirs(output_dir, exist_ok=True)

        rmse, fitness, fn, fp = [], [], [], []
        df_base = pd.read_csv(csv_path)

        for i, delay in enumerate(tqdm(time_delays, desc="Delay sweep", position=0)):
            iter_rmse, iter_fitness, iter_fp, iter_fn = [], [], [], []

            for _ in range(mc_iterations):
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

                    iter_fitness.append(pip.fitness)
                    iter_rmse.append(pip.rmse)
                    iter_fp.append(pip.fp)
                    iter_fn.append(pip.fn)

                except Exception as e:
                    log.warning(f"Iteration failed: {e}")
                    continue

            # Aggregates averages
            fitness.append(np.mean(iter_fitness) if iter_fitness else np.nan)
            rmse.append(np.mean(iter_rmse) if iter_rmse else np.nan)
            fn.append(np.mean(iter_fn) if iter_fn else np.nan)
            fp.append(np.mean(iter_fp) if iter_fp else np.nan)

        # Writes numerical arrays to disk
        archive_path = os.path.join(output_dir, "delay_sweep_metrics.npz")
        np.savez_compressed(
            archive_path,
            time_delays=time_delays,
            rmse=np.array(rmse),
            fitness=np.array(fitness),
            fn=np.array(fn),
            fp=np.array(fp),
        )

        print(f"Delay sweep metrics written successfully to {archive_path}")
        return archive_path

    def plot_saved_delay_results(
        self, archive_path, figs_output_dir="../data/figs/latest"
    ):
        """Generates figures from time delay sweep data."""
        os.makedirs(figs_output_dir, exist_ok=True)
        plt.style.use(["science", "ieee"])

        FONT_LABEL = 7
        FONT_TICK = 6
        FONT_LEG = 6

        COLOR_RMSE = "#2b5c8f"
        COLOR_FITNESS = "#d4700a"
        COLOR_FP = "#b22222"
        COLOR_FN = "#4a7c4e"

        data = np.load(archive_path)
        time_delays = data["time_delays"]
        rmse = data["rmse"]
        fitness = data["fitness"]
        fp = data["fp"]
        fn = data["fn"]

        # --- Figure 1: Geometric Alignment Metrics (Dual Y-Axes) ---
        fig1, ax1 = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

        ax1.set_xlabel(r"Time delay $\Delta t$ (s)", fontsize=FONT_LABEL)
        ax1.set_ylabel("Registration RMSE (mm)", color=COLOR_RMSE, fontsize=FONT_LABEL)
        line1 = ax1.plot(
            time_delays,
            rmse,
            color=COLOR_RMSE,
            marker="o",
            linewidth=1.2,
            markersize=3,
            label="RMSE",
        )
        ax1.tick_params(axis="y", labelcolor=COLOR_RMSE, labelsize=FONT_TICK)
        ax1.tick_params(axis="x", labelsize=FONT_TICK)
        ax1.grid(True, linestyle=":", alpha=0.4)

        # Instantiate second axes sharing the same x-spatial domain
        ax1_twin = ax1.twinx()
        ax1_twin.set_ylabel(
            "ICP alignment fitness", color=COLOR_FITNESS, fontsize=FONT_LABEL
        )
        line2 = ax1_twin.plot(
            time_delays,
            fitness,
            color=COLOR_FITNESS,
            marker="s",
            linestyle="--",
            linewidth=1.2,
            markersize=3,
            label="Fitness",
        )
        ax1_twin.tick_params(axis="y", labelcolor=COLOR_FITNESS, labelsize=FONT_TICK)

        # Gather plots from both axes to construct a consolidated legend box
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(
            lines,
            labels,
            loc="upper left",
            fontsize=FONT_LEG,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            borderpad=0.4,
        )

        fig1.savefig(
            os.path.join(figs_output_dir, "time_delay_registration.pdf"),
            bbox_inches="tight",
            dpi=300,
            transparent=True,
        )
        plt.close(fig1)

        # --- Figure 2: Damage Classification Metrics (FP & FN) ---
        fig2, ax2 = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)

        ax2.plot(
            time_delays,
            fp,
            color=COLOR_FP,
            marker="o",
            linewidth=1.2,
            markersize=3,
            label="False positives",
        )
        ax2.plot(
            time_delays,
            fn,
            color=COLOR_FN,
            marker="s",
            linestyle="--",
            linewidth=1.2,
            markersize=3,
            label="False negatives",
        )
        ax2.set_xlabel(r"Time delay $\Delta t$ (s)", fontsize=FONT_LABEL)
        ax2.set_ylabel("Classification error count", fontsize=FONT_LABEL)
        ax2.tick_params(axis="both", labelsize=FONT_TICK)
        ax2.yaxis.set_major_locator(
            MaxNLocator(integer=True)
        )  # no fractional count ticks
        ax2.grid(True, linestyle=":", alpha=0.4)
        ax2.legend(
            loc="upper left",
            fontsize=FONT_LEG,
            frameon=True,
            facecolor="white",
            edgecolor="none",
            borderpad=0.4,
        )

        fig2.savefig(
            os.path.join(figs_output_dir, "time_delay_segmentation.pdf"),
            bbox_inches="tight",
            dpi=300,
            transparent=True,
        )
        plt.close(fig2)

    def evaluate_area_accuracy(self, parquet_path="../data/damage_metrics.parquet"):
        """Loads calculated metrics, assigns nominal ground truth areas based on Y-coordinate zones, and computes prediction errors."""
        df = pd.read_parquet(parquet_path)

        # Analytical ground truth areas
        gt_areas = {
            "2.0mm": (np.pi / 4) * (2.0**2),  # ~3.141593
            "2.5mm": (np.pi / 4) * (2.5**2),  # ~4.908739
            "3.0mm": (np.pi / 4) * (3.0**2),  # ~7.068583
        }

        # Assign Ground Truth based on Y-coordinate zones
        conditions = [
            (df["centroid_y"] > -9.0),  # Highest tier (2.0mm)
            (df["centroid_y"] <= -9.0)
            & (df["centroid_y"] > -15.0),  # Middle tier (2.5mm)
            (df["centroid_y"] <= -15.0),  # Lowest tier (3.0mm)
        ]
        choices = [gt_areas["2.0mm"], gt_areas["2.5mm"], gt_areas["3.0mm"]]
        labels = ["2.0mm", "2.5mm", "3.0mm"]

        df["nominal_diameter"] = np.select(conditions, labels, default="Unknown")
        df["gt_area"] = np.select(conditions, choices, default=np.nan)

        # Compute Residuals
        df["absolute_error"] = df["projected_area"] - df["gt_area"]
        df["percent_error"] = (df["absolute_error"] / df["gt_area"]) * 100

        # First and last column of dents were not properly applied.
        df = df.iloc[3:29]

        # Display per-cluster breakdown
        print(
            "\n========================= AREA ACCURACY REPORT ========================="
        )
        print(
            f"{'ID':<5}{'Nominal':<10}{'Centroid Y':<12}{'Pred Area':<12}{'GT Area':<12}{'Error %':<10}"
        )
        print("-" * 65)
        for _, row in df.sort_values(by=["centroid_x", "centroid_y"]).iterrows():
            print(
                f"{int(row['cluster_id']):<5}"
                f"{row['nominal_diameter']:<10}"
                f"{row['centroid_y']:<12.3f}"
                f"{row['projected_area']:<12.4f}"
                f"{row['gt_area']:<12.4f}"
                f"{row['percent_error']:<+10.2f}%"
            )

        # Filter evaluation matrix for summary statistics computation
        df["absolute_error"] = df["absolute_error"][3:28]
        df["percent_error"] = df["percent_error"][3:28]

        # Summary Metrics
        mae = df["absolute_error"].abs().mean()
        mape = df["percent_error"].abs().mean()

        me = df["absolute_error"].mean()
        mpe = df["percent_error"].mean()

        print("-" * 65)
        print(f"Mean Absolute Error (MAE): {mae:.4f} mm²")
        print(f"Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        print(f"Mean Error (ME): {me:.4f} mm²")
        print(f"Mean Percentage Error (MAPE): {mpe:.2f}%")
        print(
            "========================================================================"
        )

        return df


if __name__ == "__main__":

    logging.basicConfig(
        filename="../data/monte_carlo.log",
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        force=True,
    )

    tgt_test = "../data/bus/gt/bus4.ply"
    src_test = "../data/bus/snelheid_test_bus4.ply"
    v_csv = "../data/bus/speed_files/speed_bus4.csv"

    pipe_params = {
        "velocity_scale": True,
        "downsample_reg": False,
        "sigma_thresh": 3.0,
        "percentile": 80.0,
        "damage_outliers": 200,
        "cluster_eps": 1.1,
        "cluster_min_samples": 250,
        "crop": True,
        "visualise": False,
        "write": False,
    }

    dt = DataAnalysis(pipe_params)

    # List of biases to test (e.g., odometer under-registering or over-registering)
    means = np.linspace(-0.025, 0.025, 11)

    # List of random noise levels to test
    stds = np.linspace(0.0, 0.03, 6)

    archive1 = dt.run_noise_grid_sweep(
        src_test, tgt_test, v_csv, means, stds, mc_iterations=3
    )

    # Plotting data
    # archive1 = "../data/sweep_results/sweep_metrics.npz"
    dt.plot_saved_sweep_results(archive1)

    delays = np.linspace(-0.1, 0.1, 21)

    archive2 = dt.run_noise_delay_sweep(
        src_test,
        tgt_test,
        v_csv,
        delays,
        mc_iterations=3,
    )

    # Plotting data
    # archive2 = "../data/sweep_results/delay_sweep_metrics.npz"
    dt.plot_saved_delay_results(archive2)

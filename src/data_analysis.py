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
        output_dir="../data/sweep_results",
    ):
        """Computes grid sweep and writes data to disk. No plotting occurs here."""
        os.makedirs(output_dir, exist_ok=True)
        # pc_dir = os.path.join(output_dir, "point_clouds")
        # os.makedirs(pc_dir, exist_ok=True)

        rmse_grid = np.zeros((len(stds_range), len(means_range)))
        fitness_grid = np.zeros((len(stds_range), len(means_range)))
        fp_grid = np.zeros((len(stds_range), len(means_range)))
        fn_grid = np.zeros((len(stds_range), len(means_range)))

        df_base = pd.read_csv(csv_path)

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

                rmse_grid[s_idx, m_idx] = np.mean(iter_rmse) if iter_rmse else np.nan
                fitness_grid[s_idx, m_idx] = (
                    np.mean(iter_fitness) if iter_fitness else np.nan
                )
                fp_grid[s_idx, m_idx] = np.mean(iter_fp) if iter_fp else np.nan
                fn_grid[s_idx, m_idx] = np.mean(iter_fn) if iter_fn else np.nan

        # Save all scalar grids and sweep ranges into a compressed file
        archive_path = os.path.join(output_dir, "sweep_metrics_denoise.npz")
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
        """Loads metrics from disk and outputs publication-ready IEEE two-column plots."""
        os.makedirs(figs_output_dir, exist_ok=True)
        plt.style.use(["science", "ieee"])

        # --- IEEE single-column layout constants ---
        IEEE_COL_W = 3.5  # inches — hard constraint for single-column figures
        FONT_LABEL = 7  # axis label pt
        FONT_TICK = 6  # tick labels and annotations pt
        FONT_CBAR = 6  # colorbar label pt

        data = np.load(archive_path)
        means_range = data["means_range"]
        stds_range = data["stds_range"]
        n_cols = len(means_range)
        n_rows = len(stds_range)

        # Height: preserve square cells, add a fixed margin for x-label + colorbar cap
        LABEL_MARGIN = 0.55  # inches for xlabel + colorbar top cap
        fig_h = IEEE_COL_W * (n_rows / n_cols) + LABEL_MARGIN

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
                fmt=cfg["fmt"],  # explicit format — no noisy floats
                annot_kws={"size": FONT_TICK, "weight": "regular"},
                linewidths=0.4,  # cell borders aid readability
                linecolor="white",
                square=True,  # keeps cells square regardless of fig size
                cbar_kws={
                    "label": cfg["label"],
                    "shrink": 0.65,
                    "aspect": 20,  # thinner bar = less visual weight
                    "pad": 0.02,
                },
                ax=ax,
            )

            # cbar_kws doesn't reach tick labels — must set explicitly after draw
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

            # Rotate x-labels to prevent overlap on dense axes
            ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
            ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
            ax.invert_yaxis()

            fig.savefig(
                os.path.join(figs_output_dir, cfg["file"]),
                bbox_inches="tight",
                dpi=300,  # ensures annotation text rasterises cleanly inside PDF
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
        Sweeps a 1D time delay on the velocity measurement and saves the
        numerical arrays to a compressed archive. No plotting or cluster saving occurs.
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

            # Aggregate averages safely
            fitness.append(np.mean(iter_fitness) if iter_fitness else np.nan)
            rmse.append(np.mean(iter_rmse) if iter_rmse else np.nan)
            fn.append(np.mean(iter_fn) if iter_fn else np.nan)
            fp.append(np.mean(iter_fp) if iter_fp else np.nan)

        # Serialize numerical arrays to disk
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
        """Loads raw arrays from disk and outputs publication-ready IEEE line figures."""
        os.makedirs(figs_output_dir, exist_ok=True)
        plt.style.use(["science", "ieee"])  # ieee, not just science

        # Consistent font constants (mirrors heatmap code)
        FONT_LABEL = 7
        FONT_TICK = 6
        FONT_LEG = 6

        # Clearly distinct, print-safe color pairs
        COLOR_RMSE = "#2b5c8f"  # blue
        COLOR_FITNESS = "#d4700a"  # amber — replaces "darkorange", more muted in print
        COLOR_FP = "#b22222"  # firebrick — replaces "crimson"
        COLOR_FN = "#4a7c4e"  # forest green — NOT black; survives grayscale

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

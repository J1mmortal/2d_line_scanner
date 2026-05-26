import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import seaborn as sns
from registration import Registration
from tqdm import tqdm

reg = Registration(2)


def PC_velocity_correction(V_raw, T_0, T_e, FPS, C_d, pcd):
    V = np.array(V_raw)
    T = T_e - T_0  # total time

    pcd_raw = np.asarray(pcd.points)

    # creates a list with line indices [0, 1, 2, ..., Ln]
    total_lines = int(np.floor(T * FPS)) + 1
    line_indices = np.arange(total_lines)

    # converts the first collumn of the V matrix from time to line number (negative before T_0)
    V_line_known = (V[:, 0] - T_0) * FPS
    speeds = V[:, 1]

    # linearily interpolates V_line_known to every line (auto-filters <T_0)
    V_line = np.interp(line_indices, V_line_known, speeds)
    d_line = V_line / FPS  # converts V (mm/s) to d (mm) between the lines

    # sums up the distances between lines to get the absolute y value of each line
    y_line = np.zeros(total_lines)
    y_line[1:] = np.cumsum(
        d_line[:-1]
    )  # from 1 because line 0 starts at 0, and removing final one because this is the final line

    # crops pointcloud to only contain bus (will be used later for line numbering)
    y_min = round(T_0 * FPS * C_d, 4)
    y_max = round(T_e * FPS * C_d, 4)

    PC_y_old = np.round(pcd_raw[:, 1], 4)
    bus_mask = (PC_y_old >= y_min) & (PC_y_old <= y_max)
    PC_bus = pcd_raw[bus_mask]

    # converts y values of original PC to line indices based on shared y values [0,0,0,0,1,1,1,1,1, etc.]
    _, PC_line_indices = np.unique(PC_y_old, return_inverse=True)

    # bugfix
    PC_line_indices = np.clip(PC_line_indices, 0, total_lines - 1)

    # sets y values of lines to correct calculated y values
    PC_bus[:, 1] = y_line[PC_line_indices]

    PC_corrected_o3d = o3d.geometry.PointCloud()
    PC_corrected_o3d.points = o3d.utility.Vector3dVector(PC_bus)
    PC_corrected_o3d.paint_uniform_color([1, 0.2, 0])

    # check to see wether T_0, T_e and FPS are set correctly
    # if abs(PC_line_indices[-1] - total_lines) > 2:
    #     print(
    #         "Something went wrong with the set values",
    #         ", filtered pointcloud contains {0} lines while expecting {1} lines".format(
    #             line_indices[-1], total_lines
    #         ),
    #     )

    return PC_corrected_o3d


def run_velocity_monte_carlo(
    pcd, tgt, v_raw_base, params, num_iterations=100, noise_std=0.5
):
    """
    Runs a Monte Carlo simulation by adding Gaussian noise to the measured
    velocities and evaluating the resulting registration RMSE.
    """
    rmse_results = []
    fitness_results = []

    # Extract parameters
    t_0 = params["T_0"]
    t_e = params["T_e"]
    fps = params["FPS"]
    c_d = params["C_d"]

    for i in tqdm(range(num_iterations)):
        # 1. Deep copy baseline velocity to prevent noise accumulation across loops
        v_noisy = np.array(v_raw_base, dtype=float)

        # 2. Apply Gaussian noise to the speed column (index 1)
        noise = np.random.normal(loc=0.0, scale=noise_std, size=v_noisy.shape[0])
        v_noisy[:, 1] += noise

        # 3. Execute velocity correction
        try:
            # Note: Ensure you fix the shape bug in PC_velocity_correction before calling
            pc_corrected = PC_velocity_correction(v_noisy, t_0, t_e, fps, c_d, pcd)

            # 4. Calculate registration metrics
            ransac_result = reg.get_initial_guess(pc_corrected, tgt)
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
    print(f"Min / Max RMSE:    {np.min(rmse_results):.4f} / {np.max(rmse_results):.4f}")
    print("====================================================================")
    print(f"Mean fitness:         {np.mean(fitness_results):.4f}")
    print(f"Median fitness:       {np.median(fitness_results):.4f}")
    print(f"Std Dev fitness:      {np.std(fitness_results):.4f}")
    print(
        f"Min / Max fitness:    {np.min(fitness_results):.4f} / {np.max(fitness_results):.4f}"
    )
    print("====================================================================")

    return rmse_results, fitness_results


def plot_monte_carlo_results(rmse_results, noise_std):
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
    pcd, gt_pcd, v_raw_base, params, means_range, stds_range, mc_iterations=10
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

    t_0, t_e, fps, c_d = params["T_0"], params["T_e"], params["FPS"], params["C_d"]

    # Iterate through the grid
    for s_idx, sigma in enumerate(tqdm(stds_range)):
        for m_idx, mean in enumerate(means_range):
            iter_rmse = []
            iter_fitness = []
            rsc_results = []
            pcds = []

            for _ in range(mc_iterations):
                v_noisy = np.array(v_raw_base, dtype=float)

                # Apply Gaussian noise with specific mean and std dev
                noise = np.random.normal(loc=mean, scale=sigma, size=v_noisy.shape[0])
                v_noisy[:, 1] += noise

                try:
                    pc_corrected = PC_velocity_correction(
                        v_noisy, t_0, t_e, fps, c_d, pcd
                    )

                    # 4. Calculate registration metrics
                    # ransac_result, _, _ = reg.register(pc_corrected, gt_pcd)
                    ransac_result = reg.get_initial_guess(pc_corrected, gt_pcd)

                    fitness = ransac_result.fitness
                    rmse = ransac_result.inlier_rmse

                    iter_fitness.append(fitness)
                    iter_rmse.append(rmse)

                    pc_corrected = pc_corrected.transform(ransac_result.transformation)
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
        fmt=".2f",
        cbar_kws={"label": "Mean Registration RMSE (mm)"},
    )
    plt.title("RMSE Sensitivity Matrix: Velocity Bias ($\mu$) vs. Jitter ($\sigma$)")
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
        fmt=".2f",
        cbar_kws={"label": "Mean Registration RMSE (mm)"},
    )
    plt.title("Fitness Sensitivity Matrix: Velocity Bias ($\mu$) vs. Jitter ($\sigma$)")
    plt.xlabel("Velocity Noise Mean / Bias ($\mu$ in mm/s)")
    plt.ylabel("Velocity Noise Std Dev / Jitter ($\sigma$ in mm/s)")
    plt.gca().invert_yaxis()  # Low noise at bottom, high noise at top
    plt.show()

    return rmse_grid, fitness_grid, pc_grid


# Example Integration:
# Assuming 'scores' was returned from the previous run_velocity_monte_carlo function
# plot_monte_carlo_results(scores, noise_std=1.5)

# Setup execution context
if __name__ == "__main__":
    # Define baseline inputs based on your specifications
    V_base = [
        (0, 12 * 1.0432),
        (7.999, 12 * 1.0432),
        (8, 36 * 1.0432),
        (11.999, 36 * 1.0432),
        (12, 12 * 1.0432),
        (30, 12 * 1.0432),
    ]

    sim_params = {"T_0": 1.77996, "T_e": 25.6254, "FPS": 1000, "C_d": 0.01}

    # Generate temporary dummy point clouds for script verification
    # Replace these with your actual numpy data arrays
    # np.random.seed(8)

    tgt_p = r"..\data\bus\bus_v2.ply"
    tgt = reg.load_pcd(tgt_p)
    # tgt = reg.downsample(tgt, ratio=0.001)

    src_p = r"..\data\bus\snelheid_test.ply"
    src = reg.load_pcd(src_p)
    # src = reg.downsample(src, ratio=0.001)

    # Run simulation with 50 iterations and a standard deviation of 1.5 mm/s on speed
    # rmse, fitness = run_velocity_monte_carlo(
    #     pc_raw=src_raw,
    #     tgt=tgt,
    #     v_raw_base=V_base,
    #     params=sim_params,
    #     num_iterations=50,
    #     noise_std=1.5,
    # )

    # plot_monte_carlo_results(rmse, noise_std=1.5)
    # plot_monte_carlo_results(fitness, noise_std=1.5)

    # Execution Example:
    # list of biases to test (e.g., odometer under-registering or over-registering)
    means = np.linspace(-0.3, 0.3, 5)
    # list of random noise levels to test
    stds = np.linspace(0.0, 0.3, 5)
    rmsegrid, fitnessgrid, ransacgrid = run_noise_grid_sweep(
        src, tgt, V_base, sim_params, means, stds, mc_iterations=5
    )

    topleft = ransacgrid[0, 0]
    topright = ransacgrid[0, -1]
    bottomleft = ransacgrid[-1, 0]
    bottomright = ransacgrid[-1, -1]
    o3d.visualization.draw_geometries(
        [
            reg.downsample(topleft, ratio=0.002),
            reg.downsample(tgt, ratio=0.002),
        ]
    )
    o3d.visualization.draw_geometries(
        [
            reg.downsample(topright, ratio=0.002),
            reg.downsample(tgt, ratio=0.002),
        ]
    )
    o3d.visualization.draw_geometries(
        [
            reg.downsample(bottomleft, ratio=0.002),
            reg.downsample(tgt, ratio=0.002),
        ]
    )
    o3d.visualization.draw_geometries(
        [
            reg.downsample(bottomright, ratio=0.002),
            reg.downsample(tgt, ratio=0.002),
        ]
    )

import numpy as np
import matplotlib.pyplot as plt
import open3d as o3d
import pandas as pd
import seaborn as sns
from registration import Registration
from damage_detection import DamageDetector
from tqdm import tqdm

reg = Registration(5)
det = DamageDetector()


def velocity_correction(csv_or_df, fps, C_d, pcd):
    if isinstance(csv_or_df, str):
        df = pd.read_csv(csv_or_df)
    else:
        df = csv_or_df

    v_time = df["Time_s"].values
    v_speed = df["Speed_mms"].values

    pcd_raw = np.asarray(pcd.points)

    t_0 = 1e-3 * pcd_raw[:, 1][0] / C_d
    t_e = 1e-3 * pcd_raw[:, 1][-1] / C_d

    t_tot = t_e - t_0

    # Creates a list with line indices [0, 1, 2, ..., Ln]
    total_lines = int(np.floor(t_tot * fps)) + 1
    line_indices = np.arange(total_lines)

    # Converts time to line numbers relative to T_0
    v_line_known = (v_time - t_0) * fps

    # Linearly interpolates velocity to every expected line index
    v_line = np.interp(line_indices, v_line_known, v_speed)
    d_line = v_line / fps  # Delta distance per frame

    # Cumulative sum to find absolute positions along the travel axis
    y_line = np.zeros(total_lines)
    y_line[1:] = np.cumsum(d_line[:-1])

    # Crop point cloud bounds based on nominal limits
    y_min = round(t_0 * fps * C_d, 4)
    y_max = round(t_e * fps * C_d, 4)

    pcd_y_old = np.round(pcd_raw[:, 1], 4)
    bus_mask = (pcd_y_old >= y_min) & (pcd_y_old <= y_max)
    pcd_bus = pcd_raw[bus_mask]

    # FIX: Compute line indices on the cropped subset to prevent shape mismatch
    pcd_bus_y_old = np.round(pcd_bus[:, 1], 4)
    _, PC_line_indices = np.unique(pcd_bus_y_old, return_inverse=True)

    # Bound indices to prevent out-of-bounds errors
    PC_line_indices = np.clip(PC_line_indices, 0, total_lines - 1)

    # Assign the corrected profile-derived coordinates
    pcd_bus[:, 1] = y_line[PC_line_indices]

    # Reconstruct the Open3D PointCloud object
    PC_corrected_o3d = o3d.geometry.PointCloud()
    PC_corrected_o3d.points = o3d.utility.Vector3dVector(pcd_bus)
    PC_corrected_o3d.paint_uniform_color([1, 0.2, 0])

    return PC_corrected_o3d


def velocity_correction2(csv_or_df, fps, C_d, pcd):
    if isinstance(csv_or_df, str):
        df = pd.read_csv(csv_or_df)
    else:
        df = csv_or_df

    v_time = df["Time_s"].values
    v_speed = df["Speed_mms"].values

    pcd_raw = np.asarray(pcd.points)

    # Use robust global min/max instead of assuming index 0 and -1 are sorted
    y_min_raw = pcd_raw[:, 1].min()
    y_max_raw = pcd_raw[:, 1].max()

    t_0 = 1e-3 * y_min_raw / C_d
    t_e = 1e-3 * y_max_raw / C_d
    t_tot = t_e - t_0

    # Create line indices list
    total_lines = int(np.floor(t_tot * fps)) + 1
    line_indices = np.arange(total_lines)

    # Map time to expected relative line indices
    v_line_known = (v_time - t_0) * fps
    v_line = np.interp(line_indices, v_line_known, v_speed)
    d_line = v_line / fps

    # Integrate velocity profile
    y_line = np.zeros(total_lines)
    y_line[1:] = np.cumsum(d_line[:-1])

    y_min = t_0 * fps * C_d
    y_max = t_e * fps * C_d

    pcd_y_old = pcd_raw[:, 1]
    bus_mask = (pcd_y_old >= y_min) & (pcd_y_old <= y_max)

    # Avoid slow fancy indexing if all points are within boundaries
    if np.all(bus_mask):
        pcd_bus = pcd_raw.copy()
    else:
        pcd_bus = pcd_raw[bus_mask]

    # Optimized direct arithmetic bucket mapping replaces np.unique
    PC_line_indices = np.round((pcd_bus[:, 1] - y_min) / C_d).astype(np.int32)
    np.clip(PC_line_indices, 0, total_lines - 1, out=PC_line_indices)

    # Apply profile-derived coordinates
    pcd_bus[:, 1] = y_line[PC_line_indices]

    # Reconstruct the Open3D PointCloud object
    PC_corrected_o3d = o3d.geometry.PointCloud()
    PC_corrected_o3d.points = o3d.utility.Vector3dVector(pcd_bus)
    PC_corrected_o3d.paint_uniform_color([1, 0.2, 0])

    return PC_corrected_o3d


def run_velocity_monte_carlo(
    pcd, tgt, csv_path, params, num_iterations=100, noise_std=0.5
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
            pc_corrected = velocity_correction(df_noisy, fps, c_d, pcd)

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
    pcd,
    gt_pcd,
    csv_path,
    params,
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

    t_0, t_e, fps, c_d = params["T_0"], params["T_e"], params["FPS"], params["C_d"]

    df_base = pd.read_csv(csv_path)

    if uniform_donwsample:
        tgt_reg = gt_pcd.uniform_down_sample(20)
        reg.set_voxel(tgt_reg, ratio=0.03)
    else:
        tgt_reg = reg.downsample(gt_pcd, ratio=0.001)
        reg.set_voxel(tgt_reg, ratio=0.03)

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
                    pc_corrected = velocity_correction2(df_noisy, fps, c_d, pcd)
                    # pc_corrected = det.select_bus_hull(
                    #     pc_corrected, eps=2.0, visualise=False
                    # )

                    if uniform_donwsample:
                        pc_reg = pc_corrected.uniform_down_sample(20)
                    else:
                        pc_reg = reg.downsample(pc_corrected, ratio=0.001)

                    icp, _, _ = reg.register(pc_reg, tgt_reg, ransac_retries=3)

                    # 4. Calculate registration metrics
                    # icp, _, _ = reg.register(
                    #     pc_corrected, gt_pcd, ransac_retries=3
                    # )
                    # icp = reg.get_initial_guess(pc_corrected, gt_pcd)

                    eval = reg.evaluate_alignment(
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
        fmt=".3f",
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
    T_0 = 1.77996  # time at which bus first seen by scanner (y_0 = T_0 * FPS * C_d)
    T_e = 25.6254  # time at which bus last seen by scanner (y_e = T_e * FPS * C_d)
    FPS = 1000  # internal 2D laser profiler FPS
    C_d = 0.1  # configured laser profile y distance (constant)
    sim_params = {"T_0": 1.77996, "T_e": 25.6254, "FPS": 1000, "C_d": 0.1}
    # v_csv = r"..\data\speed_test_values.csv"
    v_csv = r"..\data\bus_kinematics.csv"

    # np.random.seed(8)

    tgt_p = r"..\data\bus\bus_v2.ply"
    tgt = reg.load_pcd(tgt_p)
    # tgt = det.select_bus_hull(tgt, eps=2.0, visualise=False)
    tgt.paint_uniform_color([0.0, 1.0, 0.0])

    src_p = r"..\data\bus\snelheid_test2.ply"
    src = reg.load_pcd(src_p)

    # Run simulation with 50 iterations and a standard deviation of 1.5 mm/s on speed
    # rmse, fitness = run_velocity_monte_carlo(
    #     src,
    #     tgt=tgt,
    #     csv_path=v_csv,
    #     params=sim_params,
    #     num_iterations=50,
    #     noise_std=0.5,
    # )

    # plot_monte_carlo_results(rmse, noise_std=0.5)
    # plot_monte_carlo_results(fitness, noise_std=0.5)

    # list of biases to test (e.g., odometer under-registering or over-registering)
    means = np.linspace(-0.3, 0.3, 5)

    # list of random noise levels to test
    stds = np.linspace(0.0, 0.3, 5)

    rmsegrid, fitnessgrid, ransacgrid = run_noise_grid_sweep(
        src,
        tgt,
        v_csv,
        sim_params,
        means,
        stds,
        mc_iterations=3,
        uniform_donwsample=True,
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
    # o3d.visualization.draw_geometries([topleft, tgt])
    # o3d.visualization.draw_geometries([topright, tgt])
    # o3d.visualization.draw_geometries(bottomleft, tgt)
    # o3d.visualization.draw_geometries([bottomright, tgt])

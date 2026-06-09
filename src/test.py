from registration import Registration
from damage_detection import DamageDetector
from cloud_compare import CloudCompare
import open3d as o3d
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time

from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator
from scipy.signal import savgol_filter
from scipy.spatial import ConvexHull, cKDTree

reg = Registration(5)
det = DamageDetector()

# tgt_p = "../data/bus/bus_damagev3.ply"
# tgt_p = "../data/bus/bus_v2.ply"

tgt_p = "../data/bus/bus3.ply"
# tgt_p = "../data/bus/damage3.ply"
# tgt_p = "../data/bus/damage_80fps.ply"

src_p = "../data/bus/damage3.ply"


# def velocity_correction(csv_or_df, fps, C_d, pcd):
#     if isinstance(csv_or_df, str):
#         df = pd.read_csv(csv_or_df)
#     else:
#         df = csv_or_df

#     v_time = df["Time_s"].values
#     v_speed = df["Speed_mms"].values

#     pcd_raw = np.asarray(pcd.points)

#     # Use robust global min/max instead of assuming index 0 and -1 are sorted
#     y_min_raw = pcd_raw[:, 1].min()
#     y_max_raw = pcd_raw[:, 1].max()

#     t_0 = 1e-3 * y_min_raw / C_d
#     t_e = 1e-3 * y_max_raw / C_d
#     t_tot = t_e - t_0

#     # Create line indices list
#     total_lines = int(np.floor(t_tot * fps)) + 1
#     line_indices = np.arange(total_lines)

#     # Map time to expected relative line indices
#     v_line_known = (v_time - t_0) * fps
#     v_line = np.interp(line_indices, v_line_known, v_speed)
#     d_line = v_line / fps

#     # Integrate velocity profile
#     y_line = np.zeros(total_lines)
#     y_line[1:] = np.cumsum(d_line[:-1])

#     y_min = t_0 * fps * C_d
#     y_max = t_e * fps * C_d

#     pcd_y_old = pcd_raw[:, 1]
#     bus_mask = (pcd_y_old >= y_min) & (pcd_y_old <= y_max)

#     # Avoid slow fancy indexing if all points are within boundaries
#     if np.all(bus_mask):
#         pcd_bus = pcd_raw.copy()
#     else:
#         pcd_bus = pcd_raw[bus_mask]

#     # Optimized direct arithmetic bucket mapping replaces np.unique
#     PC_line_indices = np.round((pcd_bus[:, 1] - y_min) / C_d).astype(np.int32)
#     np.clip(PC_line_indices, 0, total_lines - 1, out=PC_line_indices)

#     # Apply profile-derived coordinates
#     pcd_bus[:, 1] = y_line[PC_line_indices]

#     # Reconstruct the Open3D PointCloud object
#     PC_corrected_o3d = o3d.geometry.PointCloud()
#     PC_corrected_o3d.points = o3d.utility.Vector3dVector(pcd_bus)
#     PC_corrected_o3d.paint_uniform_color([1, 0.2, 0])

#     return PC_corrected_o3d


# tgt = reg.load_pcd(tgt_p)
# src = reg.load_pcd(src_p)

# # src = reg.downsample(src, ratio=0.00075)
# # tgt = reg.downsample(tgt, ratio=0.00075)

# # reg.set_voxel(src, ratio=0.01)
# # print(reg.voxel)

# # reg.benchmark(src, tgt)

# csv_path = r"..\data\bus\speed_files\speed_bus4.csv"
# # csv_path = "../data/velocity_profile_halfsin.csv"

# df = pd.read_csv(csv_path)

# # time = df["time_s"].values
# # speed = df["velocity_steps_per_s"].values

# time = df["Time_s"].values
# speed = df["Speed_mms"].values

# df_noisy = df.copy()

# mean = 2.3
# sigma = 3.5

# # Apply Gaussian noise with specific mean and std dev
# noise = np.random.normal(loc=mean, scale=sigma, size=len(df_noisy))
# df_noisy["Speed_mms"] += noise

# noisy_speed = df_noisy["Speed_mms"].values

# noisy_speed = savgol_filter(noisy_speed, window_length=101, polyorder=3)


# sample_speed = speed[::50]
# sample_time = time[::50]
# noisy_sample_speed = noisy_speed[::50]


# spline = CubicSpline(sample_time, sample_speed, bc_type="natural")

# akima_spline = Akima1DInterpolator(sample_time, sample_speed)

# pchip_spline = PchipInterpolator(sample_time, sample_speed)

# spline_noisy = CubicSpline(sample_time, noisy_sample_speed, bc_type="natural")

# akima_spline_noisy = Akima1DInterpolator(sample_time, noisy_sample_speed)

# pchip_spline_noisy = PchipInterpolator(sample_time, noisy_sample_speed)

# t = np.linspace(0, max(sample_time), 1000)
# y = spline(time)
# y_a = akima_spline(time)
# y_p = pchip_spline(time)

# y_noisy = spline_noisy(time)
# y_a_noisy = akima_spline_noisy(time)
# y_p_noisy = pchip_spline_noisy(time)

# fig, ax = plt.subplots(2, 3)
# ax[0, 0].plot(time, y)
# ax[0, 0].scatter(sample_time, sample_speed)
# ax[0, 1].plot(time, y_a)
# ax[0, 1].scatter(sample_time, sample_speed)
# ax[0, 2].plot(time, y_p)
# ax[0, 2].scatter(sample_time, sample_speed)
# ax[1, 0].plot(time, y_noisy)
# ax[1, 0].scatter(sample_time, noisy_sample_speed)
# ax[1, 1].plot(time, y_a_noisy)
# ax[1, 1].scatter(sample_time, noisy_sample_speed)
# ax[1, 2].plot(time, y_p_noisy)
# ax[1, 2].scatter(sample_time, noisy_sample_speed)
# plt.show()

import numpy as np
import pandas as pd


def evaluate_area_accuracy(parquet_path="../data/damage_metrics.parquet"):
    """Loads calculated metrics, assigns nominal ground truth areas based on

    Y-coordinate zones, and computes prediction errors.
    """
    df = pd.read_parquet(parquet_path)

    # Define analytical ground truth areas
    gt_areas = {
        "2.0mm": (np.pi / 4) * (2.0**2),  # ~3.141593
        "2.5mm": (np.pi / 4) * (2.5**2),  # ~4.908739
        "3.0mm": (np.pi / 4) * (3.0**2),  # ~7.068583
    }

    # Assign Ground Truth based on Y-coordinate zones
    # Adjust these thresholds if your coordinate origin shifts
    conditions = [
        (df["centroid_y"] > -9.0),  # Highest tier (2.0mm)
        (df["centroid_y"] <= -9.0) & (df["centroid_y"] > -15.0),  # Middle tier (2.5mm)
        (df["centroid_y"] <= -15.0),  # Lowest tier (3.0mm)
    ]
    choices = [gt_areas["2.0mm"], gt_areas["2.5mm"], gt_areas["3.0mm"]]
    labels = ["2.0mm", "2.5mm", "3.0mm"]

    df["nominal_diameter"] = np.select(conditions, labels, default="Unknown")
    df["gt_area"] = np.select(conditions, choices, default=np.nan)

    # Compute Residuals
    df["absolute_error"] = df["projected_area"] - df["gt_area"]
    df["percent_error"] = (df["absolute_error"] / df["gt_area"]) * 100

    df = df.iloc[3:29]

    # Display per-cluster breakdown
    print("\n========================= AREA ACCURACY REPORT =========================")
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
    print("========================================================================")

    return df


# Execute evaluation
# df_results = evaluate_area_accuracy(parquet_path=r"..\data\bus4_gt.parquet")
df_results = evaluate_area_accuracy(parquet_path=r"..\data\damage_metrics.parquet")

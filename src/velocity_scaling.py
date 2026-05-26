import numpy as np
import open3d as o3d
import os
import sys
import pandas
import logging
import pandas as pd

from registration import Registration
from damage_detection import DamageDetector

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

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


def velocity_correctionv2(csv_or_df, fps, C_d, pcd):
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


import numpy as np
import pandas as pd
import open3d as o3d
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator


def velocity_correctionv3(
    csv_or_df, fps, C_d, pcd, method="cubic", downsample_step=100
):
    if isinstance(csv_or_df, str):
        df = pd.read_csv(csv_or_df)
    else:
        df = csv_or_df

    v_time = df["Time_s"].values
    v_speed = df["Speed_mms"].values

    pcd_raw = np.asarray(pcd.points)
    y_min_raw = pcd_raw[:, 1].min()
    y_max_raw = pcd_raw[:, 1].max()

    t_0 = 1e-3 * y_min_raw / C_d
    t_e = 1e-3 * y_max_raw / C_d
    t_tot = t_e - t_0

    total_lines = int(np.floor(t_tot * fps)) + 1
    line_indices = np.arange(total_lines)

    # Target timestamps for each discrete line profile
    t_line = t_0 + line_indices / fps

    method = method.lower()
    # Ensure there are enough points for higher-order splines
    if method in ["cubic", "akima", "pchip"] and len(v_time) > (downsample_step * 4):
        sample_time = v_time[::downsample_step]
        sample_speed = v_speed[::downsample_step]

        # Explicitly append boundary endpoints to minimize edge errors
        if sample_time[-1] != v_time[-1]:
            sample_time = np.append(sample_time, v_time[-1])
            sample_speed = np.append(sample_speed, v_speed[-1])

        # Protect against out-of-bounds NaNs and boundary oscillations
        t_line_clipped = np.clip(t_line, sample_time[0], sample_time[-1])

        if method == "cubic":
            spline_func = CubicSpline(sample_time, sample_speed, bc_type="natural")
        elif method == "akima":
            spline_func = Akima1DInterpolator(sample_time, sample_speed)
        elif method == "pchip":
            spline_func = PchipInterpolator(sample_time, sample_speed)

        v_line = spline_func(t_line_clipped)
    else:
        # Fallback to standard linear interpolation if data is too small or 'linear' is chosen
        v_line_known = (v_time - t_0) * fps
        v_line = np.interp(line_indices, v_line_known, v_speed)

    d_line = v_line / fps
    y_line = np.zeros(total_lines)
    y_line[1:] = np.cumsum(d_line[:-1])

    y_min = t_0 * fps * C_d
    y_max = t_e * fps * C_d

    pcd_y_old = pcd_raw[:, 1]
    bus_mask = (pcd_y_old >= y_min) & (pcd_y_old <= y_max)

    if np.all(bus_mask):
        pcd_bus = pcd_raw.copy()
    else:
        pcd_bus = pcd_raw[bus_mask]

    PC_line_indices = np.round((pcd_bus[:, 1] - y_min) / C_d).astype(np.int32)
    np.clip(PC_line_indices, 0, total_lines - 1, out=PC_line_indices)

    pcd_bus[:, 1] = y_line[PC_line_indices]

    PC_corrected_o3d = o3d.geometry.PointCloud()
    PC_corrected_o3d.points = o3d.utility.Vector3dVector(pcd_bus)
    PC_corrected_o3d.paint_uniform_color([1, 0.2, 0])

    return PC_corrected_o3d


if __name__ == "__main__":
    # T_0 = 1.77996  # time at which bus first seen by scanner (y_0 = T_0 * FPS * C_d)
    # T_e = 25.6254  # time at which bus last seen by scanner (y_e = T_e * FPS * C_d)
    FPS = 1000  # internal 2D laser profiler FPS
    C_d = 0.1  # configured laser profile y distance (constant)

    # Apply Gaussian noise with specific mean and std dev
    # np.random.seed(8)

    tgt_p = r"..\data\bus\bus_v2.ply"
    tgt = reg.load_pcd(tgt_p)
    # tgt = reg.downsample(tgt, ratio=0.001)
    tgt.paint_uniform_color([0, 0, 1])

    src_p = r"..\data\bus\snelheid_test2.ply"
    src = reg.load_pcd(src_p)
    # src = reg.downsample(src, ratio=0.001)

    # csv_p = r"..\data\speed_test_values.csv"
    csv_p = r"..\data\bus_kinematics.csv"

    # PC_corrected = PC_velocity_correction(v_noisy, T_0, T_e, FPS, C_d, src_raw)
    PC_corrected = velocity_correctionv3(csv_p, FPS, C_d, src)
    # print(f"Original point-cloud shape: {src_raw.shape}")
    # print(f"Corrected point_cloud shape: {PC_corrected.scale}")

    # converts the PC from the numpy array back to a downloadable o3d file
    # PC_corrected_o3d = o3d.geometry.PointCloud()
    # PC_corrected_o3d.points = o3d.utility.Vector3dVector(PC_corrected)
    # PC_corrected_o3d.paint_uniform_color([1, 0.2, 0])
    # PC_corrected = reg.downsample(PC_corrected, ratio=0.001)

    o3d.visualization.draw_geometries([reg.downsample(src, ratio=0.002)])
    o3d.visualization.draw_geometries([reg.downsample(PC_corrected, ratio=0.002)])

    icp, _, _ = reg.register(PC_corrected, tgt)
    # icp = reg.get_initial_guess(PC_corrected, tgt)
    o3d.visualization.draw_geometries(
        [
            reg.downsample(PC_corrected.transform(icp.transformation), ratio=0.002),
            reg.downsample(tgt, ratio=0.002),
        ]
    )

    bbox = tgt.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_dimension = np.max(extent)

    bbox1 = PC_corrected.get_axis_aligned_bounding_box()
    extent1 = bbox1.get_extent()
    max_dimension1 = np.max(extent1)

    print(f"ICP RMSE: {icp.inlier_rmse}, ICP fitness: {icp.fitness}")
    print(
        f"tgt max dimensions: {max_dimension}, src max dimension: {max_dimension1}. Percent difference: {(abs((max_dimension - max_dimension1)/max_dimension)*100):.2f} %"
    )

    # det.select_bus_hull(PC_corrected, eps=2.0, min_samples=10)

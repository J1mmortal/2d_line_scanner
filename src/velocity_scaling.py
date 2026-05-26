import numpy as np
import open3d as o3d
import os
import sys
import pandas
import logging

from registration import Registration

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

reg = Registration(2)

np.set_printoptions(threshold=sys.maxsize)


def PC_velocity_correction(V_raw, T_0, T_e, FPS, C_d, PC):
    V = np.array(V_raw)
    T = T_e - T_0  # total time

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

    PC_y_old = np.round(PC[:, 1], 4)
    bus_mask = (PC_y_old >= y_min) & (PC_y_old <= y_max)
    PC_bus = PC[bus_mask]

    # converts y values of original PC to line indices based on shared y values [0,0,0,0,1,1,1,1,1, etc.]
    _, PC_line_indices = np.unique(PC_y_old, return_inverse=True)

    # bugfix
    PC_line_indices = np.clip(PC_line_indices, 0, total_lines - 1)

    # sets y values of lines to correct calculated y values
    PC_bus[:, 1] = y_line[PC_line_indices]

    # check to see wether T_0, T_e and FPS are set correctly
    if abs(PC_line_indices[-1] - total_lines) > 2:
        print(
            "Something went wrong with the set values",
            ", filtered pointcloud contains {0} lines while expecting {1} lines".format(
                line_indices[-1], total_lines
            ),
        )

    return PC_bus


if __name__ == "__main__":
    T_0 = 1.77996  # time at which bus first seen by scanner (y_0 = T_0 * FPS * C_d)
    T_e = 25.6254  # time at which bus last seen by scanner (y_e = T_e * FPS * C_d)
    FPS = 1000  # internal 2D laser profiler FPS
    C_d = 0.01  # configured laser profile y distance (constant)
    # V = pandas.read_csv(r"C:\Users\roman\Documents\uni\BEP\code\test_setup\velocity_profile.csv") #measured speeds (t_i (s), V_i (mm/s))
    V = [
        (0, 12 * 1.0432),
        (7.999, 12 * 1.0432),
        (8, 36 * 1.0432),
        (11.999, 36 * 1.0432),
        (12, 12 * 1.0432),
        (30, 12 * 1.0432),
    ]
    print(V)

    v_noisy = np.array(V, dtype=float)

    # Apply Gaussian noise with specific mean and std dev
    # np.random.seed(8)
    noise = np.random.normal(loc=0.0, scale=0, size=v_noisy.shape[0])
    v_noisy[:, 1] += noise
    print(v_noisy)

    tgt_p = r"..\data\bus\bus_v2.ply"
    tgt = reg.load_pcd(tgt_p)
    tgt = reg.downsample(tgt, ratio=0.001)
    tgt.paint_uniform_color([0, 0, 1])

    src_p = r"..\data\bus\snelheid_test.ply"
    src = reg.load_pcd(src_p)
    # src = reg.downsample(src, ratio=0.001)
    src_raw = np.asarray(src.points)

    PC_corrected = PC_velocity_correction(v_noisy, T_0, T_e, FPS, C_d, src_raw)
    print(f"Original point-cloud shape: {src_raw.shape}")
    print(f"Corrected point_cloud shape: {PC_corrected.shape}")

    # converts the PC from the numpy array back to a downloadable o3d file
    PC_corrected_o3d = o3d.geometry.PointCloud()
    PC_corrected_o3d.points = o3d.utility.Vector3dVector(PC_corrected)
    PC_corrected_o3d.paint_uniform_color([1, 0.2, 0])
    PC_corrected_o3d = reg.downsample(PC_corrected_o3d, ratio=0.001)

    # downloads the pointcloud as ply
    o3d.visualization.draw_geometries([reg.downsample(src, ratio=0.002)])
    o3d.visualization.draw_geometries([reg.downsample(PC_corrected_o3d, ratio=0.002)])

    # icp, _, _ = reg.register(PC_corrected_o3d, tgt)
    icp = reg.get_initial_guess(PC_corrected_o3d, tgt)
    o3d.visualization.draw_geometries(
        [
            reg.downsample(PC_corrected_o3d.transform(icp.transformation), ratio=0.002),
            reg.downsample(tgt, ratio=0.002),
        ]
    )

    bbox = tgt.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_dimension = np.max(extent)

    bbox1 = PC_corrected_o3d.get_axis_aligned_bounding_box()
    extent1 = bbox1.get_extent()
    max_dimension1 = np.max(extent1)

    print(f"ICP RMSE: {icp.inlier_rmse}, ICP fitness: {icp.fitness}")
    print(
        f"tgt max dimensions: {max_dimension}, src max dimension: {max_dimension1}. Percent difference: {(abs((max_dimension - max_dimension1)/max_dimension)*100):.2f} %"
    )

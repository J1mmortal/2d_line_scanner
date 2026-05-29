import numpy as np
import open3d as o3d
import logging
import pandas as pd
import matplotlib.pyplot as plt

from registration import Registration
from damage_detection import DamageDetector

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

reg = Registration(5)
det = DamageDetector()


if __name__ == "__main__":

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
    PC_corrected = reg.velocity_correction(csv_p, src)
    # print(f"Original point-cloud shape: {src_raw.shape}")
    # print(f"Corrected point_cloud shape: {PC_corrected.scale}")

    # converts the PC from the numpy array back to a downloadable o3d file
    # PC_corrected_o3d = o3d.geometry.PointCloud()
    # PC_corrected_o3d.points = o3d.utility.Vector3dVector(PC_corrected)
    # PC_corrected_o3d.paint_uniform_color([1, 0.2, 0])
    # PC_corrected = reg.downsample(PC_corrected, ratio=0.001)

    o3d.visualization.draw_geometries([reg.downsample(src, ratio=0.002)])
    o3d.visualization.draw_geometries([reg.downsample(PC_corrected, ratio=0.002)])

    # icp, _, _ = reg.register(PC_corrected, tgt)
    # # icp = reg.get_initial_guess(PC_corrected, tgt)
    # o3d.visualization.draw_geometries(
    #     [
    #         reg.downsample(PC_corrected.transform(icp.transformation), ratio=0.002),
    #         reg.downsample(tgt, ratio=0.002),
    #     ]
    # )

    bbox = tgt.get_axis_aligned_bounding_box()
    extent = bbox.get_extent()
    max_dimension = np.max(extent)

    bbox1 = PC_corrected.get_axis_aligned_bounding_box()
    extent1 = bbox1.get_extent()
    max_dimension1 = np.max(extent1)

    # print(f"ICP RMSE: {icp.inlier_rmse}, ICP fitness: {icp.fitness}")
    print(
        f"tgt max dimensions: {max_dimension}, src max dimension: {max_dimension1}. Percent difference: {(abs((max_dimension - max_dimension1)/max_dimension)*100):.2f} %"
    )

    # det.select_bus_hull(PC_corrected, eps=2.0, min_samples=10)

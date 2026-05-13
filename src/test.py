from registration import Registration
from damage_detection import DamageDetector
from cloud_compare import CloudCompare
import open3d as o3d
import numpy as np

reg = Registration(10)
det = DamageDetector()


# import subprocess

# cc_path = r"C:\Program Files\CloudCompare\CloudCompare.exe"
# input = r"C:\Users\fvsch\Downloads\Dumps_face\Face_1 (2).obj"

# cmd = [
#     cc_path,
#     "-SILENT",
#     "-C_EXPORT_FMT",
#     "PCD",  # or "PLY"
#     "-NO_TIMESTAMP",
#     "-O",
#     input,
#     "-SAVE_CLOUDS",
# ]

# subprocess.run(cmd, capture_output=True, text=True, check=True)

# src_path = "../data/block_angle.ply"
# src_path = r"..\data\bus\bus_1damage.ply"
# src = reg.load_pcd(src_path).transform(reg.tf)

# tf = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

# cropped = det.crop_wheels_circular(src)
# reg.visualise_result(cropped, downsample=0.001)

# src_p = r"..\data\block\bv2_big.ply"
# tgt_p = r"..\data\block\bv2.ply"

# src_p = "../data/bus/bus_damagev2.ply"
src_p = r"..\data\CC\alg_source_CC.ply"
tgt_p = "../data/bus/bus.ply"


# ccl = CloudCompare(src_p, tgt_p)
# pcd, dist = ccl.run_cc(C2C=True)
# pcd.transform(reg.tf)
# det.visualise_colourmap(pcd, dist, downsample=0.001)

# src = reg.load_pcd(src_p).transform(reg.tf)
# src, _ = reg.SOR(src, 80, 3)

# reg.visualise_result(src, downsample=0.001)
# pln, _ = det.extract_dominant_plane(src, distance_threshold=1.12)
# # pln, _ = det.extract_dominant_plane(src, distance_threshold=0.9)
# reg.visualise_result(pln, downsample=0.001)

# # o3d.io.write_point_cloud("normal_ransac.ply", pln)
# # pln, _ = reg.SOR(pln, 20, 2)
# # pln, _ = reg.SOR(pln, 100, 1.2)
# # reg.visualise_result(pln, downsample=0.001)
# # # fpcd, _ = reg.SOR(src, 60, 3)
# fpcd, _ = reg.radius_outlier_removal(pln, 200, 1)
# # reg.visualise_result(src, downsample=0.002)
# reg.visualise_result(fpcd, downsample=0.001)

from registration import Registration
from damage_detection import DamageDetector
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
src_path = r"..\data\bus\bus_1damage.ply"
src = reg.load_pcd(src_path).transform(reg.tf)

# tf = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

cropped = det.crop_wheels_circular(src)
reg.visualise_result(cropped, downsample=0.001)


# # fpcd, _ = reg.SOR(src, 60, 3)
# fpcd, _ = reg.radius_outlier_removal(src, 60, 0.7)
# reg.visualise_result(src, downsample=0.002)
# reg.visualise_result(fpcd, downsample=0.002)

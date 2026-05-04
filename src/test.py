from registration import Registration
import open3d as o3d
import numpy as np

reg = Registration(10)

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

src_path = "../data/block_damage.ply"
src = reg.load_pcd(src_path)
fpcd, _ = reg.SOR(src, 60, 3)
reg.visualise_result(src)
reg.visualise_result(fpcd)

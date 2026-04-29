import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

from global_reg import Registration
from damage_detection import DamageDetector

reg = Registration()
det = DamageDetector()

# src = reg.load_pcd("../data/CC/sin_src.ply")
# tgt = reg.load_pcd("../data/CC/sin_tgt.ply")

dataset = o3d.data.DemoICPPointClouds()
src = o3d.io.read_point_cloud(dataset.paths[0])
tgt = o3d.io.read_point_cloud(dataset.paths[1])

# reg.set_voxel(src, ratio=0.005, coarse_ratio=0.01)

# o3d.visualization.draw_geometries([src])
# reg.visualise_result(src, tgt)

# icp, _ = reg.register(src, tgt)

# reg.visualise_result(src, tgt)
# reg.visualise_result(src, tgt, transform=icp.transformation)

# Assuming 'reg' is your Registration instance, 'src' and 'tgt' are processed clouds
ransac_result = reg.get_initial_guess(src, tgt)
init_guess = ransac_result.transformation

results = []
results.append(reg.benchmark_method(reg.icp, src, tgt, init_guess))
results.append(reg.benchmark_method(reg.plane_icp, src, tgt, init_guess))
results.append(reg.benchmark_method(reg.gen_icp, src, tgt, init_guess))

reg.print_result_summary(results)

tf = results[0]["transformation"]
reg.visualise_result(src, tgt, transform=tf, downsample=0.008)

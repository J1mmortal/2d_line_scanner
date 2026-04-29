import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

from global_reg import Registration
from damage_detection import DamageDetector

reg = Registration(voxel_size=2)
det = DamageDetector()

# src = reg.load_pcd("../data/CC/sin_src.ply")
# tgt = reg.load_pcd("../data/CC/sin_tgt.ply")

# dataset = o3d.data.DemoICPPointClouds()
# src = o3d.io.read_point_cloud(dataset.paths[0])
# tgt = o3d.io.read_point_cloud(dataset.paths[1])

# tgt = o3d.io.read_point_cloud("../data/CC/TGT.ply")
# src = o3d.io.read_point_cloud("../data/CC/SRC.ply")

src = o3d.io.read_point_cloud("../data/Reg_block_tripledented_abrupt_50000.ply")
tgt = o3d.io.read_point_cloud("../data/Reg_block_2_smooth.ply")

o3d.visualization.draw_geometries([src])
o3d.visualization.draw_geometries([tgt])


# reg.set_voxel(src, ratio=0.005, coarse_ratio=0.01)

# o3d.visualization.draw_geometries([src])
# reg.visualise_result(src, tgt)

# icp, _ = reg.register(src, tgt)

# reg.visualise_result(src, tgt)
# reg.visualise_result(src, tgt, transform=icp.transformation)

results = []

# 1. Benchmark RANSAC (Global)
global_benchmark = reg.benchmark_global_method(src, tgt)
results.append(global_benchmark)

# Extract the initial guess for the local methods
init_guess = global_benchmark["transformation"]

# 2. Benchmark ICP variants (Local)
if global_benchmark["success"]:
    results.append(reg.benchmark_method(reg.icp, src, tgt, init_guess))
    results.append(reg.benchmark_method(reg.plane_icp, src, tgt, init_guess))
    results.append(reg.benchmark_method(reg.gen_icp, src, tgt, init_guess))

# 3. Print the unified table
reg.print_result_summary(results)

tf = results[3]["transformation"]
reg.visualise_result(src, tgt, transform=tf, downsample=0.008)

alg_src

import copy
import matplotlib.pyplot as plt
import open3d as o3d

from damage_detection import DamageDetector
from global_reg import Registration

reg = Registration(course_voxel=3, voxel_size=3)
det = DamageDetector()

# f1 = "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block.stl"
# f2 = "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block_tripledented.stl"

# tgt, src = reg.poisson_convert(f1, 200000), reg.poisson_convert(f2, 200000)

tgt = o3d.io.read_point_cloud("../data/CC/sin_tgt.ply")
src = o3d.io.read_point_cloud("../data/CC/sin_src.ply")

src.paint_uniform_color([1, 0.7, 0])  # orange
tgt.paint_uniform_color([0, 0.65, 1])  # blue

# dataset = o3d.data.DemoICPPointClouds()
# src = o3d.io.read_point_cloud(dataset.paths[0])
# tgt = o3d.io.read_point_cloud(dataset.paths[1])

icp, _ = reg.register(src, tgt)

aligned_src = copy.deepcopy(src)
aligned_src.transform(icp.transformation)

# o3d.visualization.draw_geometries(
#     [src, tgt], window_name="BEFORE", width=800, height=600
# )

# o3d.visualization.draw_geometries(
#     [aligned_src, tgt], window_name="AFTER", width=800, height=600
# )


distance, _ = det.compute_bidirectional_c2c(aligned_source=aligned_src, target=tgt)

mean, std, _ = det.estimate_noise(distance, 80)

mask, distances, _ = det.detect(
    aligned_source=aligned_src, target=tgt, noise_floor=mean, noise_std=std
)

det.visualise_colourmap(aligned_src, distances=distances)

det.visualise_binary(aligned_src, mask)

labels = det.cluster(aligned_source=aligned_src, damage_mask=mask, eps=0.2)

det.color_point_cloud_by_labels(aligned_src, labels)

labels = det.cluster_fast(
    aligned_source=aligned_src, damage_mask=mask, voxel_size=0.2, eps=0.5
)

det.color_point_cloud_by_labels(aligned_src, labels)

import open3d as o3d
import copy
import numpy as np
import matplotlib.pyplot as plt

from damage_detection import DamageDetector
from global_reg import Registration

from sklearn.cluster import DBSCAN

reg = Registration(course_voxel=3, voxel_size=3)
det = DamageDetector()

f1 = "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block.stl"
f2 = "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block_tripledented.stl"

tgt, src = reg.poisson_convert(f1), reg.poisson_convert(f2)

icp, _ = reg.register(src, tgt)

aligned_src = copy.deepcopy(src)
aligned_src.transform(icp.transformation)
print(type(aligned_src))

distance, _ = det.compute_bidirectional_c2c(aligned_source=aligned_src, target=tgt)


mean, std, _ = det.estimate_noise(distance, 80)

mask, distances, _ = det.detect(
    aligned_source=aligned_src, target=tgt, noise_floor=mean, noise_std=std
)


# xyz = np.asarray(aligned_src.points)

# xyz_damage = xyz[mask]
# dist_damge = distances[mask]

# # clustering = DBSCAN(2, 10).fit((np.asarray(aligned_src), distances))

# db = DBSCAN(
#     eps=2,  # 10 cm neighborhood
#     min_samples=10,  # at least 20 nearby changed points
#     metric="euclidean",
#     n_jobs=-1,
# )

# labels = db.fit_predict(xyz_damage)

# # 3) put labels back into full arrayx
# full_labels = np.full(len(xyz), -1, dtype=int)
# full_labels[mask] = labels

labels = det.cluster(aligned_source=aligned_src, damage_mask=mask, distances=distances)

# def color_point_cloud_by_labels(
#     xyz, labels, noise_color=(0.5, 0.5, 0.5), cmap_name="tab20"
# ):
#     xyz = np.asarray(xyz, dtype=float)
#     labels = np.asarray(labels)

#     pcd = o3d.geometry.PointCloud()
#     pcd.points = o3d.utility.Vector3dVector(xyz)

#     colors = np.zeros((len(labels), 3), dtype=float)
#     unique_labels = np.unique(labels[labels >= 0])

#     if len(unique_labels) > 0:
#         cmap = plt.get_cmap(cmap_name)
#         label_to_color = {
#             lab: cmap(i / max(len(unique_labels) - 1, 1))[:3]
#             for i, lab in enumerate(unique_labels)
#         }
#         for lab, col in label_to_color.items():
#             colors[labels == lab] = col

#     colors[labels == -1] = noise_color
#     pcd.colors = o3d.utility.Vector3dVector(colors)
#     return pcd


# pcd_colored = color_point_cloud_by_labels(
#     xyz, full_labels, noise_color=(0.7, 0.7, 0.7)  # unchanged/noise = light gray
# )

print(labels, aligned_src, type(aligned_src))
det.color_point_cloud_by_labels(aligned_src, labels)

# o3d.visualization.draw_geometries([pcd_colored])
# print(set(labels), set(full_labels))

# det.visualise_binary(aligned_src, mask)

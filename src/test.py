from registration import Registration
from damage_detection import DamageDetector
from cloud_compare import CloudCompare
import open3d as o3d
import numpy as np

from scipy.spatial import ConvexHull, cKDTree

reg = Registration(10)
det = DamageDetector()

# tgt_p = "../data/bus/bus_damagev3.ply"
tgt_p = "../data/bus/bus_v2.ply"

tgt = reg.load_pcd(tgt_p)

det.select_bus_hull(tgt, eps=2.1)

# def find_largest_cluster(pcd, labels):
#     xyz = np.asarray(pcd.points)

#     all_metrics = {}
#     unique_ids = np.unique(labels)

#     # valid_labels = [lbl for lbl in unique_ids if lbl >= 0]

#     for id in unique_ids:
#         if id == -1:
#             continue

#         mask = labels == id
#         c_xyz = xyz[mask]
#         # c_dist = np.abs(distances[mask])

#         if len(c_xyz) < 3:
#             continue  # Cannot compute 2D area with < 3 points

#         # 1. PCA to find the local plane of the damage
#         mean = np.mean(c_xyz, axis=0)
#         centered = c_xyz - mean
#         cov = np.cov(centered.T)
#         evals, evecs = np.linalg.eigh(cov)

#         # evecs[:, 0] is the normal (smallest variance).
#         # evecs[:, 1:] define the flat 2D plane.
#         local_2d = centered @ evecs[:, 1:]

#         # 2. Compute exact area using Convex Hull (ignores point density)
#         try:
#             hull = ConvexHull(local_2d)
#             projected_area = float(hull.volume)  # In 2D, scipy hull.volume is the area
#             perimeter = float(hull.area)
#         except Exception as e:
#             Warning(f"Convex hull failed for cluster {id}: {e}")
#             projected_area = 0.0
#             perimeter = 0.0

#         # 3. Max depth and estimated volume
#         # max_depth = float(np.max(c_dist))
#         # avg_depth = float(np.mean(c_dist))
#         # volume = projected_area * avg_depth  # Approximated cylinder/prism volume

#         # rgb = label_to_color.get(id, (0.0, 0.0, 0.0))
#         # colour = self._get_closest_color_name(rgb)

#         all_metrics[int(id)] = {
#             "projected_area": projected_area,
#             # "volume": volume,
#             "perimeter": perimeter,
#             # "max_depth": max_depth,
#             # "color": colour,
#             # "color_rgb": rgb,
#         }

#     print(
#         "\n============================== Damage cluster metrics ================================="
#     )
#     header = f"{'Cluster ID':<12}{'Area':<14}{'Perimeter':<14}"
#     print("-" * len(header))

#     for cluster_id, data in sorted(all_metrics.items()):
#         area = f"{data['projected_area']:.6f}"
#         perimeter = f"{data['perimeter']:.6f}"

#         print(f"{cluster_id:<12}{area:<14}{perimeter:<14}")

#     return all_metrics


# find_largest_cluster(tgt, labels)

import open3d as o3d
import numpy as np
import copy
import matplotlib.pyplot as plt

from global_reg import Registration

# reg = Registration()


mesh1 = o3d.io.read_triangle_mesh(
    "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block.stl"
)
mesh1.compute_vertex_normals()

tgt = mesh1.sample_points_poisson_disk(number_of_points=10_000)
tgt.paint_uniform_color([0, 0.65, 1])

mesh2 = o3d.io.read_triangle_mesh(
    "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block_dented.stl"
)
mesh2.compute_vertex_normals()

src = mesh2.sample_points_poisson_disk(number_of_points=10_000)
src.paint_uniform_color([1, 0.7, 0])


# o3d.visualization.draw_geometries([tgt, src], window_name="Block")

reg = Registration(3, 3)
icp_result, global_result = reg.register(src, tgt)

tf = icp_result.transformation
print(f"ICP RMSE: {icp_result.inlier_rmse}, RANSCAC RMSE: {global_result.inlier_rmse}")

src.paint_uniform_color([1, 0.7, 0])  # orange
tgt.paint_uniform_color([0, 0.65, 1])  # blue

alg_src = copy.deepcopy(src)
alg_src.transform(tf)

# Show before and after in two windows
# o3d.visualization.draw_geometries(
#     [src, tgt], window_name="BEFORE", width=800, height=600
# )

o3d.visualization.draw_geometries(
    [alg_src, tgt], window_name="AFTER", width=800, height=600
)


damage_pcd = alg_src.compute_point_cloud_distance(tgt)
distances = np.asarray(damage_pcd)

fig, axes = plt.subplots(1, 2, figsize=(13, 4))

ax = axes[0]
ax.hist(distances, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
ax.axvline(np.median(distances), color="gray", linestyle="--", label="median")
ax.set_xlabel("C2C distance (mm)")
ax.set_ylabel("Point count")
ax.set_title("Full C2C distance distribution")
ax.legend()

# Zoom into the bulk (noise floor) to show its shape
ax2 = axes[1]
bulk = distances[(5 < distances) & (distances < 10)]
ax2.hist(bulk, bins=60, color="mediumseagreen", edgecolor="none", alpha=0.8)
ax2.set_xlabel("C2C distance (mm)")
ax2.set_title("Bulk (noise floor region, < 3 mm)")
ax2.set_ylabel("Point count")

plt.tight_layout()
# plt.savefig('/tmp/c2c_histogram.png', dpi=150, bbox_inches='tight')
plt.show()

bulk_cutoff = np.percentile(distances, 80)
bulk_dists = distances[distances < bulk_cutoff]

noise_mean = float(bulk_dists.mean())
noise_std = float(bulk_dists.std())

N_SIGMA = 3.0
threshold = noise_mean + N_SIGMA * noise_std

damage_mask = distances > threshold
n_damaged = damage_mask.sum()

aligned_pts = np.asarray(alg_src.points)

# Filter to the side panel (Y ≈ 150 mm) for the clearest view
side_mask = np.abs(aligned_pts[:, 0]) > 30
x_side = aligned_pts[side_mask, 0]
z_side = aligned_pts[side_mask, 2]
d_side = distances[side_mask]
dmg_side = damage_mask[side_mask]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Continuous distance map
ax = axes[0]
sc = ax.scatter(
    x_side,
    z_side,
    c=d_side,
    cmap="RdYlBu_r",
    s=1,
    vmin=0,
    vmax=np.percentile(d_side, 99),
)
plt.colorbar(sc, ax=ax, label="C2C distance (mm)")
ax.set_xlabel("X — longitudinal (mm)")
ax.set_ylabel("Z — height (mm)")
ax.set_title("Damage map — continuous C2C distance (side panel)")
ax.set_aspect("equal")

# Binary damage mask
ax2 = axes[1]
ax2.scatter(
    x_side[~dmg_side], z_side[~dmg_side], c="lightgray", s=0.8, label="undamaged"
)
ax2.scatter(x_side[dmg_side], z_side[dmg_side], c="tomato", s=2, label="damage flagged")
ax2.set_xlabel("X — longitudinal (mm)")
ax2.set_ylabel("Z — height (mm)")
ax2.set_title(f"Binary damage mask (threshold = {threshold:.2f} mm = {N_SIGMA}σ)")
ax2.legend(markerscale=6)
ax2.set_aspect("equal")

plt.tight_layout()
# plt.savefig('/tmp/damage_map.png', dpi=150, bbox_inches='tight')
plt.show()

import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Normalise distances to [0, 1] for colormap
vmax = np.percentile(distances, 99)  # cap outliers
norm = mcolors.Normalize(vmin=0, vmax=vmax)
cmap = cm.get_cmap("RdYlBu_r")  # red = high distance = damage

# Map each point's distance to an RGB colour
colors_rgb = cmap(norm(distances))[:, :3]  # shape (N, 3), drop alpha

# Apply to the aligned source cloud
vis_pcd = copy.deepcopy(alg_src)
vis_pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

o3d.visualization.draw_geometries(
    [vis_pcd], window_name="C2C Damage Heatmap", width=1200, height=800
)

colors = np.where(
    damage_mask[:, None],  # broadcast over RGB
    [1.0, 0.1, 0.1],  # red = damaged
    [0.75, 0.75, 0.75],  # grey = undamaged
)

vis_pcd = copy.deepcopy(alg_src)
vis_pcd.colors = o3d.utility.Vector3dVector(colors)

o3d.visualization.draw_geometries(
    [vis_pcd], window_name="Binary Damage Mask 3D", width=1200, height=800
)

import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy

# from sklearn.cluster import DBSCAN
from scipy.spatial import cKDTree


class DamageDetector:
    def __init__(
        self,
        damage_sigma_threshold=3.0,
        dbscan_eps=2.0,
        dbscan_min_points=30,
        keep_largest_cluster=False,
    ):
        self.damage_sigma_threshold = damage_sigma_threshold
        self.dbscan_eps = dbscan_eps
        self.dbscan_min_points = dbscan_min_points
        self.keep_largest_cluster = keep_largest_cluster

    def downsample(self, pcd, voxel_ratio=0.008, normal_max_nn=30):
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_dimension = np.max(extent)

        # Set voxel size to X% of the largest dimension (e.g., 0.01 = 1%)
        dynamic_voxel = max_dimension * voxel_ratio

        # Downsample
        pcd_down = pcd.voxel_down_sample(voxel_size=dynamic_voxel)

        # Estimate normals. radius must scale with the dynamic voxel.
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=dynamic_voxel * 2.5,
                max_nn=normal_max_nn,
            )
        )
        return pcd_down

    def detect(self, aligned_source, target, noise_floor, noise_std):
        pcd_dist = aligned_source.compute_point_cloud_distance(target)
        distances = np.asarray(pcd_dist)

        threshold = noise_floor + self.damage_sigma_threshold * noise_std
        damage_mask = distances > threshold

        return damage_mask, distances, pcd_dist

    def cluster(
        self, aligned_source, damage_mask, eps=2.0, min_samples=10
    ):  # EPS parameter is very important, must be chosen to match data magnitude and point density
        xyz = np.asarray(aligned_source.points)
        xyz_damage = xyz[damage_mask]

        # Handle edge case where no damage is found
        if len(xyz_damage) == 0:
            return np.full(len(xyz), -1, dtype=int)

        # Push damaged points into a temporary Open3D object
        damage_pcd = o3d.geometry.PointCloud()
        damage_pcd.points = o3d.utility.Vector3dVector(xyz_damage)

        # Execute C++ optimized DBSCAN
        labels = np.asarray(
            damage_pcd.cluster_dbscan(
                eps=eps, min_points=min_samples, print_progress=True
            )
        )

        # Map labels back to the full array
        full_labels = np.full(len(xyz), -1, dtype=int)
        full_labels[damage_mask] = labels

        return full_labels

    def cluster_fast(
        self, aligned_source, damage_mask, voxel_size=0.5, eps=2.0, min_samples=10
    ):
        xyz = np.asarray(aligned_source.points)
        xyz_damage = xyz[damage_mask]

        if len(xyz_damage) == 0:
            return np.full(len(xyz), -1, dtype=int)

        # 1. Isolate damage and downsample
        damage_pcd = o3d.geometry.PointCloud()
        damage_pcd.points = o3d.utility.Vector3dVector(xyz_damage)

        # voxel_size must be scaled to your coordinate units
        down_pcd = damage_pcd.voxel_down_sample(voxel_size=voxel_size)
        down_xyz = np.asarray(down_pcd.points)

        # 2. Cluster the lightweight downsampled cloud
        # print_progress=True will prove if the algorithm is actually hanging
        labels_down = np.asarray(
            down_pcd.cluster_dbscan(
                eps=eps, min_points=min_samples, print_progress=True
            )
        )

        # 3. Map labels back to the dense damage points using a KDTree
        # This finds the nearest downsampled point for every dense point and copies its label
        tree = cKDTree(down_xyz)
        _, indices = tree.query(xyz_damage, k=1)
        labels_dense = labels_down[indices]

        # 4. Reconstruct the full label array
        full_labels = np.full(len(xyz), -1, dtype=int)
        full_labels[damage_mask] = labels_dense

        return full_labels

    def color_point_cloud_by_labels(
        self, aligned_source, labels, noise_color=(0.5, 0.5, 0.5), cmap_name="tab20"
    ):
        xyz = np.asarray(aligned_source.points)
        labels = np.asarray(labels)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(xyz)

        colors = np.zeros((len(labels), 3), dtype=float)
        unique_labels = np.unique(labels[labels >= 0])

        if len(unique_labels) > 0:
            cmap = plt.get_cmap(cmap_name)
            label_to_color = {
                lab: cmap(i / max(len(unique_labels) - 1, 1))[:3]
                for i, lab in enumerate(unique_labels)
            }
            for lab, col in label_to_color.items():
                colors[labels == lab] = col

        colors[labels == -1] = noise_color
        pcd.colors = o3d.utility.Vector3dVector(colors)

        o3d.visualization.draw_geometries([self.downsample(pcd)])

    def estimate_noise(self, distances, percentile=80):
        bulk_cutoff = np.percentile(distances, percentile)
        bulk_dists = distances[distances < bulk_cutoff]

        noise_mean = float(bulk_dists.mean())
        noise_std = float(bulk_dists.std())

        threshold = noise_mean + self.damage_sigma_threshold * noise_std

        return noise_mean, noise_std, threshold

    def compute_bidirectional_c2c(self, aligned_source, target):
        src_to_tgt = np.asarray(
            aligned_source.compute_point_cloud_distance(target), dtype=float
        )
        tgt_to_src = np.asarray(
            target.compute_point_cloud_distance(aligned_source), dtype=float
        )
        return src_to_tgt, tgt_to_src

    def plot_distance_hist(self, distances, percentile=80):
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
        bulk = distances[distances < np.percentile(distances, percentile)]
        ax2.hist(bulk, bins=60, color="mediumseagreen", edgecolor="none", alpha=0.8)
        ax2.set_xlabel("C2C distance (mm)")
        ax2.set_title(
            f"Bulk (noise floor region, < {np.percentile(distances, percentile)})"
        )
        ax2.set_ylabel("Point count")

        plt.tight_layout()
        # plt.savefig('/tmp/c2c_histogram.png', dpi=150, bbox_inches='tight')
        plt.show()

    def visualise_binary(self, pcd, damage_mask):
        colors = np.where(
            damage_mask[:, None],  # broadcast over RGB
            [1.0, 0.1, 0.1],  # red = damaged
            [0.75, 0.75, 0.75],  # grey = undamaged
        )

        vis_pcd = copy.deepcopy(pcd)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)

        pcd = self.downsample(vis_pcd)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Binary Damage Mask 3D",
            width=1200,
            height=800,
        )

    def visualise_colourmap(self, pcd, distances):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # Normalise distances to [0, 1] for colormap
        vmax = np.percentile(distances, 99)  # cap outliers
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap = cm.get_cmap("RdYlBu_r")  # red = high distance = damage

        # Map each point's distance to an RGB colour
        colors_rgb = cmap(norm(distances))[:, :3]  # shape (N, 3), drop alpha

        # Apply to the aligned source cloud
        vis_pcd = copy.deepcopy(pcd)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        pcd = self.downsample(vis_pcd)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="C2C Damage Heatmap",
            width=1200,
            height=800,
        )

    # Need to correctly pass the damage plane as coming from the scanner
    def calculate_damage_metrics(
        self, pcd, distances, labels, target_cluster_id, grid_res=0.25
    ):
        """
        Calculates 2.5D metrics for a specific damage cluster.
        grid_res MUST be in the same physical units as your XYZ coordinates.
        """
        # 1. Isolate the target cluster
        xyz = np.asarray(pcd.points)

        mask = labels == target_cluster_id
        c_xyz = xyz[mask]
        c_dist = np.abs(distances[mask])  # Ensure distances are positive magnitudes

        if len(c_xyz) == 0:
            raise ValueError(f"Cluster {target_cluster_id} contains no points.")

        # 2. Shift coordinates to local origin for grid indexing
        # Assumption: The damage is roughly aligned to the XY plane.
        x_local = c_xyz[:, 0] - np.min(c_xyz[:, 0])
        y_local = c_xyz[:, 2] - np.min(c_xyz[:, 2])

        # 3. Convert coordinates to integer grid indices
        x_idx = (x_local / grid_res).astype(int)
        y_idx = (y_local / grid_res).astype(int)

        # 4. Initialize the 2D raster grid
        max_x, max_y = np.max(x_idx), np.max(y_idx)
        grid = np.zeros((max_x + 1, max_y + 1))

        # 5. Populate the grid.
        # If multiple laser points fall in the same cell, we take the maximum damage depth.
        np.maximum.at(grid, (x_idx, y_idx), c_dist)

        # --- METRIC CALCULATIONS ---

        # Area
        cell_area = grid_res**2
        active_cells = np.count_nonzero(grid)
        projected_area = active_cells * cell_area

        # Volume (Sum of depth * area for all cells)
        volume = np.sum(grid) * cell_area

        # Max Depth
        max_depth = np.max(grid)

        return {
            "cluster_id": target_cluster_id,
            "projected_area": projected_area,
            "volume": volume,
            "max_depth": max_depth,
            "grid_matrix": grid,  # Returned so you can slice cross-sections
        }

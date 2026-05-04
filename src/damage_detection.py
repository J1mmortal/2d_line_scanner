import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings

from scipy.spatial import ConvexHull, cKDTree
from scipy.signal import medfilt, wiener


class DamageDetector:
    def __init__(self):
        self.damage_sigma_threshold = None

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

    def detect(
        self,
        aligned_source,
        target,
        sigma_thresh=3,
        percentile=80,
        median_filter_kernel=None,
    ):
        pcd_dist = aligned_source.compute_point_cloud_distance(target)
        distances = np.asarray(pcd_dist)

        mean, std, threshold = self.estimate_noise(
            distances, percentile=percentile, sigma_thresh=sigma_thresh
        )

        # Median (or wiener) filter to smooth noise
        if median_filter_kernel is not None:
            distances = medfilt(distances, median_filter_kernel)
            # distances = wiener(distances, median_filter_kernel)

        damage_mask = distances > threshold

        return damage_mask, distances

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

        bbox = aligned_source.get_axis_aligned_bounding_box()
        max_dim = np.max(bbox.get_extent())
        if eps > max_dim * 0.1:
            warnings.warn(
                f"eps={eps} is large relative to cloud extent ({max_dim:.2f}). Check units."
            )
        if eps < max_dim * 0.001:
            warnings.warn(
                f"eps={eps} is small relative to cloud extent ({max_dim:.2f}). Check units."
            )

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

    def estimate_noise(self, distances, percentile, sigma_thresh):
        bulk_cutoff = np.percentile(distances, percentile)
        bulk_dists = distances[distances < bulk_cutoff]

        noise_mean = float(bulk_dists.mean())
        noise_std = float(bulk_dists.std())

        threshold = noise_mean + sigma_thresh * noise_std

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
            width=1600,
            height=1000,
        )

    def visualise_colourmap(self, pcd, distances):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # Normalise distances to [0, 1] for colormap
        vmax = np.percentile(distances, 99)  # cap outliers
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap = cm.get_cmap("RdYlBu_r")  # red = high distance = damage
        # cmap = cm.get_cmap("turbo")

        # Map each point's distance to an RGB colour
        colors_rgb = cmap(norm(distances))[:, :3]  # shape (N, 3), drop alpha

        # Apply to the aligned source cloud
        vis_pcd = copy.deepcopy(pcd)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        pcd = self.downsample(vis_pcd)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Damage Heatmap",
            width=1600,
            height=1000,
        )

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

        o3d.visualization.draw_geometries(
            [self.downsample(pcd)],
            window_name=f"Damage clustered into {len(set(labels))-1} regions",
            width=1600,
            height=1000,
        )

    # Need to correctly pass the damage plane as coming from the scanner
    def calculate_damage_metrics(
        self, pcd, distances, labels, grid_res=0.25, cmap_name="tab20"
    ):
        """
        Calculates 2.5D metrics for a specific damage cluster.
        grid_res MUST be in the same physical units as your XYZ coordinates.
        """
        # 1. Isolate the target cluster
        xyz = np.asarray(pcd.points)

        all_metrics = {}
        unique_ids = np.unique(labels)

        valid_labels = [lbl for lbl in unique_ids if lbl >= 0]
        label_to_color = {}
        if len(valid_labels) > 0:
            cmap = plt.get_cmap(cmap_name)
            label_to_color = {
                lab: cmap(i / max(len(valid_labels) - 1, 1))[:3]
                for i, lab in enumerate(valid_labels)
            }

        for id in unique_ids:
            if id == -1:
                continue

            mask = labels == id
            c_xyz = xyz[mask]
            c_dist = np.abs(distances[mask])

            if len(c_xyz) < 3:
                continue  # Cannot compute 2D area with < 3 points

            # 1. PCA to find the local plane of the damage
            mean = np.mean(c_xyz, axis=0)
            centered = c_xyz - mean
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)

            # evecs[:, 0] is the normal (smallest variance).
            # evecs[:, 1:] define the flat 2D plane.
            local_2d = centered @ evecs[:, 1:]

            # 2. Compute exact area using Convex Hull (ignores point density)
            try:
                hull = ConvexHull(local_2d)
                projected_area = float(
                    hull.volume
                )  # In 2D, scipy hull.volume is the area
                perimeter = float(hull.area)
            except Exception as e:
                Warning(f"Convex hull failed for cluster {id}: {e}")
                projected_area = 0.0
                perimeter = 0.0

            # 3. Max depth and estimated volume
            max_depth = float(np.max(c_dist))
            avg_depth = float(np.mean(c_dist))
            volume = projected_area * avg_depth  # Approximated cylinder/prism volume

            rgb = label_to_color.get(id, (0.0, 0.0, 0.0))
            colour = self._get_closest_color_name(rgb)

            all_metrics[int(id)] = {
                "projected_area": projected_area,
                "volume": volume,
                "perimeter": perimeter,
                "max_depth": max_depth,
                "color": colour,
                "color_rgb": rgb,
            }

        print(
            "\n============================== Damage cluster metrics ================================="
        )
        header = f"{'Cluster ID':<12}{'Area':<14}{'Volume':<14}{'Perimeter':<14}{'Max Depth':<14}{'Color (R, G, B)':<20}"
        print(header)
        print("-" * len(header))

        for cluster_id, data in sorted(all_metrics.items()):
            area = f"{data['projected_area']:.6f}"
            volume = f"{data['volume']:.6f}"
            perimeter = f"{data['perimeter']:.6f}"
            depth = f"{data['max_depth']:.6f}"

            colour = data["color"]
            r, g, b = data["color_rgb"]
            color_str = f"{colour} ({r:.2f}, {g:.2f}, {b:.2f})"

            print(
                f"{cluster_id:<12}{area:<14}{volume:<14}{perimeter:<14}{depth:<14}{color_str:<20}"
            )

        return all_metrics

    def _get_closest_color_name(self, rgb):
        """Finds the closest human-readable color name for an RGB tuple."""
        # Standard colors mapped to RGB in the 0.0 - 1.0 range
        named_colors = {
            "Red": (1.0, 0.0, 0.0),
            "Dark Red": (0.5, 0.0, 0.0),
            "Green": (0.0, 0.5, 0.0),
            "Lime": (0.0, 1.0, 0.0),
            "Light Green": (0.6, 0.98, 0.6),
            "Blue": (0.0, 0.0, 1.0),
            "Navy": (0.0, 0.0, 0.5),
            "Light Blue": (0.68, 0.85, 0.9),
            "Yellow": (1.0, 1.0, 0.0),
            "Gold": (1.0, 0.84, 0.0),
            "Cyan": (0.0, 1.0, 1.0),
            "Teal": (0.0, 0.5, 0.5),
            "Magenta": (1.0, 0.0, 1.0),
            "Purple": (0.5, 0.0, 0.5),
            "Orange": (1.0, 0.65, 0.0),
            "Dark Orange": (1.0, 0.55, 0.0),
            "Pink": (1.0, 0.75, 0.8),
            "Deep Pink": (1.0, 0.08, 0.58),
            "Brown": (0.65, 0.16, 0.16),
            "Maroon": (0.5, 0.0, 0.0),
            "Gray": (0.5, 0.5, 0.5),
            "Silver": (0.75, 0.75, 0.75),
            "Black": (0.0, 0.0, 0.0),
            "White": (1.0, 1.0, 1.0),
        }

        min_dist = float("inf")
        closest_name = "Unknown"

        for name, target_rgb in named_colors.items():
            # Calculate squared Euclidean distance
            dist = sum((a - b) ** 2 for a, b in zip(rgb, target_rgb))
            if dist < min_dist:
                min_dist = dist
                closest_name = name

        return closest_name

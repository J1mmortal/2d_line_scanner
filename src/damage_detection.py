import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
import logging
from sklearn.neighbors import KDTree

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

    def detect_damage(
        self,
        aligned_source,
        target,
        sigma_thresh=3,
        percentile=80,
        bidirectional=True,
        remove_outliers=False,
    ):
        pcd_dist = aligned_source.compute_point_cloud_distance(target)
        src_distances = np.asarray(pcd_dist)

        mean, std, threshold = self.estimate_noise(
            src_distances, percentile, sigma_thresh
        )
        damage_mask = src_distances > threshold

        if bidirectional and damage_mask.sum() > 0:
            tgt_dist = target.compute_point_cloud_distance(aligned_source)
            tgt_distances = np.asarray(tgt_dist)

            # For each flagged src point, find its nearest tgt point's reverse distance
            src_pts = np.asarray(aligned_source.points)[damage_mask]
            tgt_pts = np.asarray(target.points)

            tree = cKDTree(tgt_pts)
            _, nn_idx = tree.query(src_pts, k=1)
            reverse_dists = tgt_distances[nn_idx]

            mean_r, std_r, threshold_r = self.estimate_noise(
                tgt_distances, percentile, sigma_thresh
            )

            # Only keep damage that is anomalous in BOTH directions
            confirmed = reverse_dists > threshold_r

            bidirectional_mask = np.zeros_like(damage_mask)
            bidirectional_mask[np.where(damage_mask)[0][confirmed]] = True
            damage_mask = bidirectional_mask

        if remove_outliers and damage_mask.sum() > 0:
            damaged_indices = np.where(damage_mask)[0]
            damaged_pcd = aligned_source.select_by_index(damaged_indices)
            clean_pcd, valid_inliers = damaged_pcd.remove_radius_outlier(
                nb_points=100, radius=4
            )
            clean_damage_mask = np.zeros_like(damage_mask)
            clean_damage_mask[damaged_indices[valid_inliers]] = True
            damage_mask = clean_damage_mask

        return damage_mask, src_distances

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
            return warnings.warn(
                f"eps={eps} is large relative to cloud extent ({max_dim:.2f}). Check units."
            )

        if eps < max_dim * 0.001:
            return warnings.warn(
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
        self, aligned_source, damage_mask=None, voxel_size=0.5, eps=2.0, min_samples=10
    ):
        xyz = np.asarray(aligned_source.points)
        if damage_mask is not None:
            xyz_damage = xyz[damage_mask]

        xyz_damage = xyz

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

    def crop_wheels_circular(self, pcd):
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        extent = bbox.get_extent()

        points = np.asarray(pcd.points)

        # 1. Define wheel center locations (relative to bounding box)
        # Adjust these percentages to match your specific bus model
        rel_front_x = 0.24  # Front axle at 20% of bus length
        rel_rear_x = 0.735  # Rear axle at 80% of bus length
        rel_y = 0.13  # Hub centers at 12% of bus height

        # 2. Map relative centers to absolute world coordinates
        front_cx = min_bound[0] + (rel_front_x * extent[0])
        rear_cx = min_bound[0] + (rel_rear_x * extent[0])
        cz = min_bound[1] + (rel_y * extent[1])

        # 3. Define the physical radius of the crop region
        # Using a fraction of the bus height (e.g., 15%) keeps it scale-invariant
        radius = 0.19 * extent[1]
        radius_sq = radius**2

        # 4. Calculate squared Euclidean distance in the 2D XZ plane
        # We ignore the Y axis because we want to cut a cylinder completely through the bus
        dist_to_front_sq = (points[:, 0] - front_cx) ** 2 + (points[:, 1] - cz) ** 2
        dist_to_rear_sq = (points[:, 0] - rear_cx) ** 2 + (points[:, 1] - cz) ** 2

        # 5. Create boolean masks for points inside the circles
        is_front_wheel = dist_to_front_sq < radius_sq
        is_rear_wheel = dist_to_rear_sq < radius_sq

        wheel_mask = is_front_wheel | is_rear_wheel

        # 6. Invert mask to keep the rest of the bus
        keep_mask = ~wheel_mask
        valid_indices = np.where(keep_mask)[0]

        return pcd.select_by_index(valid_indices)

    def extract_dominant_plane(self, pcd, distance_threshold=1.05):
        """
        Uses RANSAC to find the main face (dominant plane) even if slightly tilted.
        """
        # Find the largest plane in the point cloud
        plane_model, inliers = pcd.segment_plane(
            distance_threshold=distance_threshold, ransac_n=3, num_iterations=1000
        )
        removed = len(pcd.points) - len(pcd.select_by_index(inliers).points)

        # Return only the points that belong to that plane
        return pcd.select_by_index(inliers), removed

    def select_bus_hull(
        self, pcd, voxel_size=1, eps=2.0, min_samples=10, visualise=True
    ):
        labels = self.cluster_fast(
            pcd, voxel_size=voxel_size, eps=eps, min_samples=min_samples
        )

        if visualise:
            self.color_point_cloud_by_labels(pcd, labels)

        xyz = np.asarray(pcd.points)
        cropped_xyz = xyz[labels == 0]

        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(cropped_xyz)

        if visualise:
            cropped_pcd.paint_uniform_color([0.0, 0.0, 1.0])
            o3d.visualization.draw_geometries(
                [self.downsample(cropped_pcd)],
                window_name="Bus Hull",
                width=1600,
                height=1000,
            )

        return cropped_pcd

    def crop_damage(
        self,
        pcd,
        mask,
        max_y_threshold,
        x_thresh,
        height_axis=1,
        width_axis=0,
        robust_floor=True,
    ):
        if max_y_threshold is None:
            return mask

        xyz = np.asarray(pcd.points)
        heights = xyz[:, height_axis]
        widths = xyz[:, width_axis]

        # Calculate floor relative to actual point distribution
        floor_y = np.percentile(heights, 1) if robust_floor else np.min(heights)
        abs_thresh_y = floor_y + max_y_threshold

        floor_x = np.percentile(widths, 1) if robust_floor else np.min(widths)
        roof_x = np.percentile(widths, 99) if robust_floor else np.max(widths)
        min_thresh_x = floor_x + x_thresh
        max_thresh_x = roof_x - x_thresh

        # Create spatial mask and intersect with damage mask
        valid_height_mask = heights <= abs_thresh_y
        valid_width_mask = (min_thresh_x <= widths) & (widths <= max_thresh_x)
        filtered_mask = mask & valid_height_mask & valid_width_mask

        removed_count = mask.sum() - filtered_mask.sum()
        y_name = ["X", "Y", "Z"][height_axis]
        x_name = ["X", "Y", "Z"][width_axis]

        logging.info(
            "Height and width filter (Rel %s: %.2fm; %s: %.2fm) removed %d points.",
            y_name,
            max_y_threshold,
            x_name,
            x_thresh,
            removed_count,
        )

        return filtered_mask

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

    def visualise_binary(self, pcd, damage_mask, downsample=0.008, write=False):
        colors = np.where(
            damage_mask[:, None],  # broadcast over RGB
            [1.0, 0.1, 0.1],  # red = damaged
            [0.75, 0.75, 0.75],  # grey = undamaged
        )

        vis_pcd = copy.deepcopy(pcd)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)

        if write:
            o3d.io.write_point_cloud(
                "../data/debug/binary_cloud.ply",
                vis_pcd,
            )

        pcd = self.downsample(vis_pcd, voxel_ratio=downsample)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Binary Damage Mask 3D",
            width=1600,
            height=1000,
        )

    def visualise_colourmap(self, pcd, distances, downsample=0.008, write=False):
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        # Normalise distances to [0, 1] for colormap
        vmax = np.percentile(distances, 98)  # cap outliers
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap = cm.get_cmap("turbo")  # red = high distance = damage
        # cmap = cm.get_cmap("turbo")

        # Map each point's distance to an RGB colour
        colors_rgb = cmap(norm(distances))[:, :3]  # shape (N, 3), drop alpha

        # Apply to the aligned source cloud
        vis_pcd = copy.deepcopy(pcd)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        if write:
            o3d.io.write_point_cloud(
                "../data/debug/cmap_cloud.ply",
                vis_pcd,
            )

        pcd = self.downsample(vis_pcd, voxel_ratio=downsample)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Damage Heatmap",
            width=1600,
            height=1000,
        )

    def color_point_cloud_by_labels(
        self,
        aligned_source,
        labels,
        noise_color=(0.5, 0.5, 0.5),
        cmap_name="tab20",
        downsample=0.008,
        write=False,
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

        if write:
            o3d.io.write_point_cloud(
                "../data/debug/clustered_cloud.ply",
                pcd,
            )

        o3d.visualization.draw_geometries(
            [self.downsample(pcd, voxel_ratio=downsample)],
            window_name=f"Damage clustered into {len(set(labels))-1} regions",
            width=1600,
            height=1000,
        )

    # Need to correctly pass the damage plane as coming from the scanner
    def calculate_damage_metrics(self, pcd, distances, labels, cmap_name="tab20"):
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

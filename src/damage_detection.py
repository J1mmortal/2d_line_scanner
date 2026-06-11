import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
import warnings
import logging
from sklearn.neighbors import KDTree
import pandas as pd

from scipy.spatial import ConvexHull, cKDTree
from scipy.signal import medfilt, wiener
from scipy.spatial import convex_hull_plot_2d
from scipy.stats import zscore


# Damage detection pipeline: C2C distance thresholding against a MAD noise floor,
# DBSCAN clustering of candidate regions, and 2.5D metric extraction per cluster.
class DamageDetector:
    def __init__(self):
        self.damage_sigma_threshold = None

    def downsample(self, pcd, voxel_ratio=0.008, normal_max_nn=30):
        """Downsamples point cloud based on ratio of max dimension."""
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_dimension = np.max(extent)

        dynamic_voxel = max_dimension * voxel_ratio
        pcd_down = pcd.voxel_down_sample(voxel_size=dynamic_voxel)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=dynamic_voxel * 2.5,
                max_nn=normal_max_nn,
            )
        )
        return pcd_down

    def select_bus_hull(
        self, pcd, voxel_size=1, eps=2.0, min_samples=10, visualise=True
    ):
        """Segments the bus hull from the full scene cloud via DBSCAN; returns the largest cluster."""
        labels = self.cluster_fast(
            pcd, voxel_size=voxel_size, eps=eps, min_samples=min_samples
        )

        if visualise:
            self.color_point_cloud_by_labels(pcd, labels)

        # Noise points carry label -1; the bus hull is the largest valid cluster
        valid_labels = labels[labels != -1]
        if len(valid_labels) == 0:
            raise RuntimeError("Invalid clustering, check eps and min_samples values")

        largest_label = np.bincount(valid_labels).argmax()

        xyz = np.asarray(pcd.points)
        cropped_xyz = xyz[labels == largest_label]

        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(cropped_xyz)

        if visualise:
            cropped_pcd.paint_uniform_color([0.0, 0.0, 1.0])
            o3d.visualization.draw_geometries(
                [self.downsample(cropped_pcd, voxel_ratio=0.002)],
                window_name="Bus Hull",
                width=1600,
                height=1000,
            )

        return cropped_pcd

    # --- Damage detection ---

    def estimate_noise_mad(self, distances, sigma_thresh=3.0):
        """Robust noise floor estimation via MAD; assumes >50% of points represent true surface overlap."""
        noise_median = np.median(distances)
        mad = np.median(np.abs(distances - noise_median))

        # Scale factor maps MAD to equivalent Gaussian standard deviation
        noise_std = 1.4826 * mad
        threshold = noise_median + sigma_thresh * noise_std

        return float(noise_median), float(noise_std), float(threshold)

    def detect_damage(
        self,
        aligned_source,
        target,
        sigma_thresh=3,
        percentile=80,
        bidirectional=True,
        outliers_points=None,
        radius=2,
    ):
        """
        Identifies candidate damage points via C2C distance thresholding.
        Bidirectional check retains only points anomalous in both directions,
        suppressing false positives from surface offset rather than true dents.
        Optional ROR pass removes isolated noise points from the final mask.
        """
        pcd_dist = aligned_source.compute_point_cloud_distance(target)
        src_distances = np.asarray(pcd_dist)

        mean, std, threshold = self.estimate_noise_mad(src_distances, sigma_thresh)
        damage_mask = src_distances > threshold

        if bidirectional and damage_mask.sum() > 0:
            tgt_dist = target.compute_point_cloud_distance(aligned_source)
            tgt_distances = np.asarray(tgt_dist)

            # For each flagged source point find its nearest target point's reverse distance
            src_pts = np.asarray(aligned_source.points)[damage_mask]
            tgt_pts = np.asarray(target.points)

            tree = cKDTree(tgt_pts)
            _, nn_idx = tree.query(src_pts, k=1)
            reverse_dists = tgt_distances[nn_idx]

            # NOTE: noise threshold should be estimated from tgt_distances here,
            # not src_distances — using src_distances produces an inconsistent threshold
            # for the reverse direction check.
            mean_r, std_r, threshold_r = self.estimate_noise_mad(
                src_distances, sigma_thresh
            )

            confirmed = reverse_dists > threshold_r

            bidirectional_mask = np.zeros_like(damage_mask)
            bidirectional_mask[np.where(damage_mask)[0][confirmed]] = True
            damage_mask = bidirectional_mask

        if outliers_points is not None and damage_mask.sum() > 0:
            damaged_indices = np.where(damage_mask)[0]
            damaged_pcd = aligned_source.select_by_index(damaged_indices)
            clean_pcd, valid_inliers = damaged_pcd.remove_radius_outlier(
                nb_points=outliers_points, radius=radius
            )

            logging.info(
                f"Removed {len(damaged_pcd.points) - len(valid_inliers)} points from damage mask using "
                f"Radius Outlier Removal with nb_points = {outliers_points} and radius = {radius}"
            )

            clean_damage_mask = np.zeros_like(damage_mask)
            clean_damage_mask[damaged_indices[valid_inliers]] = True
            damage_mask = clean_damage_mask

        return damage_mask, src_distances, threshold

    # --- Clustering ---

    def cluster(
        self, aligned_source, damage_mask, eps=2.0, min_samples=10, verbose=True
    ):
        """DBSCAN clustering on the full-density damage point set. eps must match point density and data units."""
        xyz = np.asarray(aligned_source.points)
        xyz_damage = xyz[damage_mask]

        if len(xyz_damage) == 0:
            return np.full(len(xyz), -1, dtype=int)

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
                eps=eps, min_points=min_samples, print_progress=verbose
            )
        )

        full_labels = np.full(len(xyz), -1, dtype=int)
        full_labels[damage_mask] = labels
        return full_labels

    def cluster_fast(
        self, aligned_source, damage_mask=None, voxel_size=0.5, eps=2.0, min_samples=10
    ):
        """
        Faster DBSCAN variant: voxel-downsamples the damage set before clustering,
        then propagates labels back to the full-density cloud via nearest-neighbour lookup.
        """
        xyz = np.asarray(aligned_source.points)
        xyz_damage = xyz if damage_mask is None else xyz[damage_mask]

        if len(xyz_damage) == 0:
            return np.full(len(xyz), -1, dtype=int)

        damage_pcd = o3d.geometry.PointCloud()
        damage_pcd.points = o3d.utility.Vector3dVector(xyz_damage)

        down_pcd = damage_pcd.voxel_down_sample(voxel_size=voxel_size)
        down_xyz = np.asarray(down_pcd.points)

        labels_down = np.asarray(
            down_pcd.cluster_dbscan(
                eps=eps, min_points=min_samples, print_progress=False
            )
        )

        # Propagate downsampled labels to dense points via nearest-neighbour lookup
        tree = cKDTree(down_xyz)
        _, indices = tree.query(xyz_damage, k=1)
        labels_dense = labels_down[indices]

        full_labels = np.full(len(xyz), -1, dtype=int)
        full_labels[damage_mask] = labels_dense
        return full_labels

    # --- Geometric cropping helpers ---

    # NOTE: superseded by select_bus_hull, which is geometry-agnostic and doesn't
    # require hardcoded axle positions. Retained in case wheel removal is needed
    # as a post-segmentation step.
    def crop_wheels_circular(self, pcd):
        """Crops out wheels using hardcoded relative axle positions and a circular radius mask."""
        bbox = pcd.get_axis_aligned_bounding_box()
        min_bound = bbox.get_min_bound()
        extent = bbox.get_extent()

        points = np.asarray(pcd.points)

        rel_front_x = 0.24
        rel_rear_x = 0.735
        rel_y = 0.13

        front_cx = min_bound[0] + (rel_front_x * extent[0])
        rear_cx = min_bound[0] + (rel_rear_x * extent[0])
        cz = min_bound[1] + (rel_y * extent[1])

        radius = 0.19 * extent[1]
        radius_sq = radius**2

        dist_to_front_sq = (points[:, 0] - front_cx) ** 2 + (points[:, 1] - cz) ** 2
        dist_to_rear_sq = (points[:, 0] - rear_cx) ** 2 + (points[:, 1] - cz) ** 2

        is_front_wheel = dist_to_front_sq < radius_sq
        is_rear_wheel = dist_to_rear_sq < radius_sq

        wheel_mask = is_front_wheel | is_rear_wheel
        keep_mask = ~wheel_mask
        valid_indices = np.where(keep_mask)[0]

        return pcd.select_by_index(valid_indices)

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
        """Trims the damage mask spatially to exclude bumpers and window height regions."""
        if max_y_threshold is None:
            return mask

        xyz = np.asarray(pcd.points)
        heights = xyz[:, height_axis]
        widths = xyz[:, width_axis]

        floor_y = np.percentile(heights, 1) if robust_floor else np.min(heights)
        abs_thresh_y = floor_y + max_y_threshold

        floor_x = np.percentile(widths, 1) if robust_floor else np.min(widths)
        roof_x = np.percentile(widths, 99) if robust_floor else np.max(widths)
        min_thresh_x = floor_x + x_thresh
        max_thresh_x = roof_x - x_thresh

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

    # --- Metrics ---

    def calculate_damage_metrics(
        self, pcd, distances, labels, cmap_name="tab20", write_to_pd=False, verbose=True
    ):
        """
        Computes 2.5D metrics per damage cluster: projected area and perimeter via
        convex hull on the PCA-flattened cluster plane, plus max/mean depth and an
        approximate prism volume. Optionally writes results to Parquet.
        """
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
                continue

            # PCA to find the best-fit plane; project onto the two principal axes
            # to get a 2D footprint for hull and area computation
            mean = np.mean(c_xyz, axis=0)
            centered = c_xyz - mean
            cov = np.cov(centered.T)
            evals, evecs = np.linalg.eigh(cov)

            local_2d = centered @ evecs[:, 1:]
            local_2d = self.filter_spatial_outliers(local_2d, 2.5)

            try:
                hull = ConvexHull(local_2d)
                projected_area = float(
                    hull.volume
                )  # scipy uses hull.volume for 2D area
                perimeter = float(hull.area)

                if verbose:
                    self.visualize_hull(local_2d, hull, id)
            except Exception as e:
                Warning(f"Convex hull failed for cluster {id}: {e}")
                projected_area = 0.0
                perimeter = 0.0

            max_depth = float(np.max(c_dist))
            avg_depth = float(np.mean(c_dist))
            volume = (
                projected_area * avg_depth
            )  # Prism approximation: area × mean depth

            rgb = label_to_color.get(id, (0.0, 0.0, 0.0))
            colour = self._get_closest_color_name(rgb)

            all_metrics[int(id)] = {
                "centroid": mean,
                "projected_area": projected_area,
                "volume": volume,
                "perimeter": perimeter,
                "max_depth": max_depth,
                "color": colour,
                "color_rgb": rgb,
            }

        if verbose:
            print(
                "\n=========================== Damage cluster metrics =============================="
            )
            header = f"{'Cluster ID':<12}{'Location':<35}{'Area':<14}{'Volume':<14}{'Perimeter':<14}{'Max Depth':<14}{'Color (R, G, B)':<20}"
            print(header)
            print("-" * len(header))

            for cluster_id, data in sorted(all_metrics.items()):
                centroid = np.array2string(
                    data["centroid"], precision=3, separator=", ", floatmode="fixed"
                )
                area = f"{data['projected_area']:.6f}"
                volume = f"{data['volume']:.6f}"
                perimeter = f"{data['perimeter']:.6f}"
                depth = f"{data['max_depth']:.6f}"
                colour = data["color"]
                r, g, b = data["color_rgb"]
                color_str = f"{colour} ({r:.2f}, {g:.2f}, {b:.2f})"
                print(
                    f"{cluster_id:<12}{centroid:<35}{area:<14}{volume:<14}{perimeter:<14}{depth:<14}{color_str:<20}"
                )

        if write_to_pd:
            rows = []
            for cluster_id, data in all_metrics.items():
                centroid = data["centroid"]
                row = {
                    "cluster_id": int(cluster_id),
                    "centroid_x": float(centroid[0]),
                    "centroid_y": float(centroid[1]),
                    "centroid_z": float(centroid[2]) if len(centroid) > 2 else 0.0,
                    "projected_area": float(data["projected_area"]),
                    "volume": float(data["volume"]),
                    "perimeter": float(data["perimeter"]),
                    "max_depth": float(data["max_depth"]),
                }
                rows.append(row)

            df = pd.DataFrame(rows)
            df.to_parquet(
                "../data/damage_metrics.parquet", engine="pyarrow", index=False
            )

        return all_metrics

    def filter_spatial_outliers(self, local_2d, threshold=2.5):
        """Removes stray points from a projected cluster using Z-score on distance to centroid."""
        distances = np.linalg.norm(local_2d, axis=1)
        z_scores = zscore(distances)
        return local_2d[np.abs(z_scores) < threshold]

    def compare_cluster_runs(
        self,
        gt_parquet_path: str,
        guessed_parquet_path: str,
        max_distance: float,
        compact_view=False,
        verbose=False,
    ):
        """Scores predicted cluster locations against ground truth via spatial radius matching; returns FP and FN counts."""
        df_gt = pd.read_parquet(gt_parquet_path)
        df_guess = pd.read_parquet(guessed_parquet_path)

        coord_cols = ["centroid_x", "centroid_y", "centroid_z"]
        gt_coords = df_gt[coord_cols].to_numpy()
        guess_coords = df_guess[coord_cols].to_numpy()

        gt_tree = cKDTree(gt_coords)
        guess_tree = cKDTree(guess_coords)

        # A predicted cluster is a true positive if any GT cluster lies within max_distance
        guess_matches = gt_tree.query_ball_point(guess_coords, r=max_distance)
        df_guess["match_status"] = [
            "Success" if len(matches) > 0 else "False Positive"
            for matches in guess_matches
        ]

        # A GT cluster is a false negative if no prediction falls within max_distance
        gt_matches = guess_tree.query_ball_point(gt_coords, r=max_distance)
        df_gt["false_negative"] = [len(matches) == 0 for matches in gt_matches]

        match_len = len(df_guess["match_status"])
        total_matches = (df_guess["match_status"] == "Success").sum()
        total_fp = (df_guess["match_status"] == "False Positive").sum()
        total_fn = (df_gt["false_negative"] == True).sum()

        if verbose:
            logging.info(
                f"Number of False Positives: {total_fp}. Number of False Negatives: {total_fn}"
            )

            if compact_view:
                if total_fn > 0 and total_fp > 0:
                    df_guess_fp = df_guess.loc[
                        df_guess["match_status"] == "False Positive"
                    ].reset_index(drop=True)
                    df_gt_fn = df_gt.loc[df_gt["false_negative"] == True].reset_index(
                        drop=True
                    )
                    logging.info(
                        f"Guess\n{df_guess_fp}\n\n{'=' * 50} Ground truth {'=' * 50}\n{df_gt_fn}"
                    )
                elif total_fn == 0 and total_fp > 0:
                    df_guess_fp = df_guess.loc[
                        df_guess["match_status"] == "False Positive"
                    ]
                    logging.info(f"Guess \n{df_guess_fp}")
                elif total_fn > 0 and total_fp == 0:
                    df_gt_fn = df_gt.loc[df_gt["false_negative"] == True]
                    logging.info(f"Ground truth\n{df_gt_fn}")
            else:
                logging.info(
                    f'\n{"=" * 58} Guess {"=" * 58}\n{df_guess}\n\n'
                    f'{"=" * 58} Ground truth {"=" * 58}\n{df_gt}'
                )

        return total_fp, total_fn

    def _get_closest_color_name(self, rgb):
        """Maps an RGB tuple to the nearest human-readable color name by squared Euclidean distance."""
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
            dist = sum((a - b) ** 2 for a, b in zip(rgb, target_rgb))
            if dist < min_dist:
                min_dist = dist
                closest_name = name
        return closest_name

    # --- Visualisation ---

    def visualize_hull(self, local_2d, hull, cluster_id):
        """Plots the 2D projected cluster points and their convex hull for visual verification."""
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(
            local_2d[:, 0],
            local_2d[:, 1],
            "o",
            color="blue",
            markersize=3,
            label="Projected Points",
        )

        for simplex in hull.simplices:
            ax.plot(local_2d[simplex, 0], local_2d[simplex, 1], "r-", linewidth=2)

        ax.plot(
            local_2d[hull.vertices, 0],
            local_2d[hull.vertices, 1],
            "ro",
            markersize=6,
            label="Hull Vertices",
        )
        ax.set_title(
            f"Cluster {cluster_id} - Convex Hull Verification. Number of points: {len(local_2d)}"
        )
        ax.set_xlabel("Local X' (PCA Eigenvector 1)")
        ax.set_ylabel("Local Y' (PCA Eigenvector 2)")
        ax.axis("equal")
        ax.grid(True, linestyle="--", alpha=0.6)
        ax.legend()
        plt.savefig(f"../data/figs/hulls/hull_{cluster_id}")

    def plot_distance_hist(self, distances, percentile=80):
        """Plots full and bulk-only C2C distance histograms for noise floor inspection."""
        fig, axes = plt.subplots(1, 2, figsize=(13, 4))

        ax = axes[0]
        ax.hist(distances, bins=100, color="steelblue", edgecolor="none", alpha=0.8)
        ax.axvline(np.median(distances), color="gray", linestyle="--", label="median")
        ax.set_xlabel("C2C distance (mm)")
        ax.set_ylabel("Point count")
        ax.set_title("Full C2C distance distribution")
        ax.legend()

        ax2 = axes[1]
        bulk = distances[distances < np.percentile(distances, percentile)]
        ax2.hist(bulk, bins=60, color="mediumseagreen", edgecolor="none", alpha=0.8)
        ax2.set_xlabel("C2C distance (mm)")
        ax2.set_title(
            f"Bulk (noise floor region, < {np.percentile(distances, percentile)})"
        )
        ax2.set_ylabel("Point count")

        plt.tight_layout()
        plt.show()

    def visualise_binary(self, pcd, damage_mask, downsample=0.008, write=False):
        """Colours the point cloud red/grey by binary damage mask for quick visual inspection."""
        colors = np.where(
            damage_mask[:, None],
            [1.0, 0.1, 0.1],
            [0.75, 0.75, 0.75],
        )

        vis_pcd = copy.deepcopy(pcd)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors)

        if write:
            o3d.io.write_point_cloud("../data/debug/binary_cloud.ply", vis_pcd)

        pcd = self.downsample(vis_pcd, voxel_ratio=downsample)
        o3d.visualization.draw_geometries(
            [pcd],
            window_name="Binary Damage Mask 3D",
            width=1600,
            height=1000,
        )

    def visualise_colourmap(self, pcd, distances, downsample=0.008, write=False):
        """Colours the point cloud by C2C distance using a turbo heatmap; high distance = likely damage."""
        import matplotlib.cm as cm
        import matplotlib.colors as mcolors

        vmax = np.percentile(distances, 98)
        norm = mcolors.Normalize(vmin=0, vmax=vmax)
        cmap = cm.get_cmap("turbo")

        colors_rgb = cmap(norm(distances))[:, :3]

        vis_pcd = copy.deepcopy(pcd)
        vis_pcd.colors = o3d.utility.Vector3dVector(colors_rgb)

        if write:
            o3d.io.write_point_cloud("../data/debug/cmap_cloud.ply", vis_pcd)

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
        noise_color=(0.2, 0.2, 0.2),
        cmap_name="tab20",
        downsample=0.008,
        write=False,
        vis=True,
    ):
        """Colours the point cloud by cluster label; noise points (-1) rendered in dark grey."""
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
            o3d.io.write_point_cloud("../data/debug/clustered_cloud.ply", pcd)

        if vis:
            o3d.visualization.draw_geometries(
                [self.downsample(pcd, voxel_ratio=downsample)],
                window_name=f"Damage clustered into {len(set(labels))-1} regions",
                width=1600,
                height=1000,
            )

        return self.downsample(pcd, voxel_ratio=downsample)

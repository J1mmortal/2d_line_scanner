import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy
from sklearn.cluster import DBSCAN


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

    def detect(self, aligned_source, target, noise_floor, noise_std):
        pcd_dist = aligned_source.compute_point_cloud_distance(target)
        distances = np.asarray(pcd_dist)

        threshold = noise_floor + self.damage_sigma_threshold * noise_std
        damage_mask = distances > threshold

        return damage_mask, distances, pcd_dist

    def cluster(self, aligned_source, damage_mask, distances, eps=2, min_samples=10):
        xyz = np.asarray(aligned_source.points)

        xyz_damage = xyz[damage_mask]
        # dist_damge = distances[damage_mask]

        db = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric="euclidean",
            n_jobs=-1,
        )

        labels = db.fit_predict(xyz_damage)

        full_labels = np.full(len(xyz), -1, dtype=int)
        full_labels[damage_mask] = labels

        return full_labels

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

        o3d.visualization.draw_geometries(
            [vis_pcd], window_name="Binary Damage Mask 3D", width=1200, height=800
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

        o3d.visualization.draw_geometries(
            [vis_pcd], window_name="C2C Damage Heatmap", width=1200, height=800
        )

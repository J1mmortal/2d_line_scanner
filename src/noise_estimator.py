import open3d as o3d
import numpy as np
import matplotlib.pyplot as plt
import copy


# Maybe useful later on, saved as an idea
class NoiseFloorEstimator:
    """
    Estimate baseline noise from repeated 'healthy' scans of the same object.
    Assumes scans are already roughly in the same pose, or a registration
    function is provided.
    """

    def __init__(self, bulk_percentile=80, sigma_multiplier=3.0, use_robust=True):
        self.bulk_percentile = bulk_percentile
        self.sigma_multiplier = sigma_multiplier
        self.use_robust = use_robust

        self.noise_floor = None
        self.noise_std = None
        self.threshold = None
        self.scan_stats = []

    @staticmethod
    def _robust_stats(x):
        x = np.asarray(x)
        med = np.median(x)
        mad = np.median(np.abs(x - med))
        sigma = 1.4826 * mad
        if sigma == 0:
            sigma = float(np.std(x))
        return float(med), float(sigma)

    @staticmethod
    def _classic_stats(x):
        x = np.asarray(x)
        return float(np.mean(x)), float(np.std(x))

    def _bulk_region(self, distances):
        cutoff = np.percentile(distances, self.bulk_percentile)
        bulk = distances[distances <= cutoff]
        return bulk, float(cutoff)

    def fit(self, reference_pcd, baseline_scans, register_fn=None):
        """
        reference_pcd: point cloud used as reference geometry
        baseline_scans: list of 'healthy' scans
        register_fn: optional function(scan, reference_pcd) -> transformed_scan
                     or -> registration result with .transformation
        """
        all_bulk = []
        self.scan_stats = []

        for i, scan in enumerate(baseline_scans):
            src = copy.deepcopy(scan)

            if register_fn is not None:
                reg_out = register_fn(src, reference_pcd)

                if hasattr(reg_out, "transformation"):
                    src.transform(reg_out.transformation)
                elif isinstance(reg_out, tuple) and hasattr(
                    reg_out[0], "transformation"
                ):
                    src.transform(reg_out[0].transformation)
                elif isinstance(reg_out, o3d.geometry.PointCloud):
                    src = reg_out
                else:
                    raise ValueError(
                        "register_fn must return a point cloud or result with .transformation"
                    )

            dists = np.asarray(src.compute_point_cloud_distance(reference_pcd))
            bulk, cutoff = self._bulk_region(dists)

            if self.use_robust:
                mu, sigma = self._robust_stats(bulk)
            else:
                mu, sigma = self._classic_stats(bulk)

            self.scan_stats.append(
                {
                    "scan_idx": i,
                    "n_points": len(dists),
                    "bulk_cutoff": cutoff,
                    "noise_floor": mu,
                    "noise_std": sigma,
                    "median_dist": float(np.median(dists)),
                    "p95_dist": float(np.percentile(dists, 95)),
                }
            )

            all_bulk.append(bulk)

        all_bulk = np.concatenate(all_bulk)

        if self.use_robust:
            self.noise_floor, self.noise_std = self._robust_stats(all_bulk)
        else:
            self.noise_floor, self.noise_std = self._classic_stats(all_bulk)

        self.threshold = self.noise_floor + self.sigma_multiplier * self.noise_std
        return self.noise_floor, self.noise_std, self.threshold

    def summary(self):
        if self.threshold is None:
            raise RuntimeError("Call fit() first.")
        return {
            "noise_floor": self.noise_floor,
            "noise_std": self.noise_std,
            "threshold": self.threshold,
            "n_baseline_scans": len(self.scan_stats),
        }

    def plot_baseline_distributions(
        self, reference_pcd, baseline_scans, register_fn=None
    ):
        fig, ax = plt.subplots(figsize=(8, 4.5))

        for i, scan in enumerate(baseline_scans):
            src = copy.deepcopy(scan)

            if register_fn is not None:
                reg_out = register_fn(src, reference_pcd)
                if hasattr(reg_out, "transformation"):
                    src.transform(reg_out.transformation)
                elif isinstance(reg_out, tuple) and hasattr(
                    reg_out[0], "transformation"
                ):
                    src.transform(reg_out[0].transformation)
                elif isinstance(reg_out, o3d.geometry.PointCloud):
                    src = reg_out

            dists = np.asarray(src.compute_point_cloud_distance(reference_pcd))
            ax.hist(dists, bins=100, alpha=0.25, density=True, label=f"scan {i}")

        if self.threshold is not None:
            ax.axvline(
                self.threshold,
                color="red",
                linestyle="--",
                label=f"threshold = {self.threshold:.3f}",
            )

        ax.set_xlabel("C2C distance")
        ax.set_ylabel("Density")
        ax.set_title("Baseline scan distance distributions")
        ax.legend()
        plt.tight_layout()
        plt.show()


# # healthy_scans = [scan1, scan2, scan3, scan4]
# # reg.register(src, tgt) -> (icp_result, global_result)

# def register_to_ref(scan, ref):
#     icp_result, global_result = reg.register(scan, ref)
#     aligned = copy.deepcopy(scan)
#     aligned.transform(icp_result.transformation)
#     return aligned

# noise_est = NoiseFloorEstimator(
#     bulk_percentile=80,
#     sigma_multiplier=3.0,
#     use_robust=True
# )

# noise_floor, noise_std, threshold = noise_est.fit(
#     reference_pcd=tgt,
#     baseline_scans=healthy_scans,
#     register_fn=register_to_ref
# )

# print(noise_est.summary())

# detector = DamageDetector()
# damage_mask, distances, _ = detector.detect(
#     aligned_source=alg_src,
#     target=tgt,
#     noise_floor=noise_floor,
#     noise_std=noise_std
# )

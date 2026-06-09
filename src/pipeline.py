import copy
import logging
import yaml
import numpy as np
import open3d as o3d

from registration import Registration
from damage_detection import DamageDetector
from cloud_compare import CloudCompare

log = logging.getLogger(__name__)
# logging.basicConfig(
#     filename="../data/run.log",
#     level=logging.INFO,
#     format="%(asctime)s [%(levelname)s] %(message)s",
# )


class Pipeline:
    def __init__(
        self,
        source_path: str,
        target_path: str,
        src_pcd=None,
        tgt_pcd=None,
        select_hull: bool = True,
        velocity_scale: bool = False,
        speed_data=None,
        sor_neighbours: int = None,
        sor_std: float = 1.0,
        voxel_size: float = 5.0,
        downsample_reg: bool = False,
        sigma_thresh: float = 3.0,
        percentile: float = 80.0,
        crop: bool = True,
        damage_outliers=None,
        damage_radius=2,
        cluster_eps: float = 2.0,
        cluster_min_samples: int = 10,
        fast_cluster: bool = False,
        min_fitness: float = 0.5,
        visualise: bool = True,
        benchmark=False,
        cc=False,
        c2c=True,
        m3c2=False,
        aligned_path="../data/CC/alg_source_CC.ply",
        tgt_path="../data/CC/tgt_CC.ply",
        skip_reg=False,
        write: bool = False,
    ):
        self.reg = Registration(voxel_size)
        self.det = DamageDetector()
        self.ccl = CloudCompare(comp_path=source_path, ref_path=target_path)

        self.sor_neighbours = sor_neighbours
        self.sor_std = sor_std
        self.select_hull = select_hull
        self.velocity_scale = velocity_scale

        self.downsample_reg = downsample_reg

        self.sigma_thresh = sigma_thresh
        self.percentile = percentile
        self.crop = crop
        self.damage_outliers = damage_outliers
        self.damage_radius = damage_radius

        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.fast_cluster = fast_cluster
        self.min_fitness = min_fitness
        self.visualise = visualise
        self.benchmark = benchmark
        self.cc = cc
        self.c2c = c2c
        self.m3c2 = m3c2
        self.skip_reg = skip_reg
        self.write = write

        self.aligned_path = aligned_path
        self.tgt_path = tgt_path
        self.gt_parquet_path = "../data/bus4_gt.parquet"
        self.guess_parquet_path = "../data/damage_metrics.parquet"

        self.src = src_pcd if src_pcd is not None else self.reg.load_pcd(source_path)
        self.tgt = tgt_pcd if tgt_pcd is not None else self.reg.load_pcd(target_path)

        self.tgt = self.tgt.transform(self.reg.tf)

        if self.velocity_scale:
            log.info("Scaling bus point cloud based on velocity...")
            if speed_data is None:
                self.src = self.reg.velocity_correction_cont(
                    "../data/bus/speed_files/speed_bus4.csv",
                    self.src,
                    denoise=False,
                    downsample_step=20,
                    visualise=False,
                )
            else:
                self.src = self.reg.velocity_correction(
                    speed_data, self.src, visualise=False
                )

        self.src = self.src.transform(self.reg.tf)

        if self.select_hull:
            if not skip_reg:
                log.info("Segmenting bus hull...")
                self.src = self.det.select_bus_hull(self.src, eps=2.1, visualise=False)
                self.tgt = self.det.select_bus_hull(self.tgt, eps=2.05, visualise=False)

        if self.sor_neighbours is not None:
            log.info("Removing statistical outliers...")
            self.src, removed = self.reg.SOR(
                self.src, self.sor_neighbours, self.sor_std
            )
            self.tgt, _ = self.reg.SOR(self.tgt, self.sor_neighbours, self.sor_std)
            log.info(f"Performed Statistical Outlier Removal, removed {removed} points")

        # Results populated by run()
        self.alg_src = None
        self.tgt_reg = None
        self.src_reg = None
        self.cropped_pcd = None
        self.transformation = None
        self.mask = None
        self.distances = None
        self.labels = None
        self.metrics = None
        self.rmse = None
        self.fitness = None
        self.fp = None
        self.fn = None
        self.n_clusters = None

    def run(self):
        # self.reg.set_voxel(self.tgt)
        log.info(f"Number of points in point cloud: {len(self.tgt.points)}")

        if self.benchmark:
            self._benchmark()
            return log.info("Benchmarking complete")

        if not self.skip_reg:
            self._register()
            self._detect()
            self._cluster()
            self._compute_metrics()
        else:
            self.alg_src = self.reg.load_pcd(self.aligned_path)
            self.tgt = self.reg.load_pcd(self.tgt_path)
            self._detect()
            self._cluster()
            self._compute_metrics()
        return self.metrics

    def _register(self):
        log.info("Starting registration...")

        if self.downsample_reg:
            self.tgt_reg = self.tgt.uniform_down_sample(15)
            self.reg.set_voxel(self.tgt_reg, ratio=0.03)

            self.src_reg = self.src.uniform_down_sample(15)

            icp, _, _ = self.reg.register(self.src_reg, self.tgt_reg, ransac_retries=5)

            eval = self.reg.evaluate_alignment(self.src, self.tgt, icp.transformation)

            log.info("ICP fitness: %.4f  RMSE: %.6f", eval.fitness, eval.inlier_rmse)

            self.rmse = eval.inlier_rmse
            self.fitness = eval.fitness

            self.transformation = icp.transformation
            self.alg_src = copy.deepcopy(self.src).transform(self.transformation)
        else:
            icp, _, eval = self.reg.register(self.src, self.tgt)

            self.rmse = eval.inlier_rmse
            self.fitness = eval.fitness

            log.info("ICP fitness: %.4f  RMSE: %.6f", eval.fitness, eval.inlier_rmse)

            if eval.fitness < self.min_fitness:
                raise RuntimeError(
                    f"Registration fitness {eval.fitness:.3f} is below threshold "
                    f"{self.min_fitness}. Check inputs or voxel size."
                )

            self.transformation = icp.transformation
            self.alg_src = copy.deepcopy(self.src).transform(self.transformation)

        if self.visualise:
            self.reg.visualise_result(
                self.src,
                self.tgt,
                self.transformation,
                downsample=0.001,
                write=self.write,
            )

    def _detect(self):
        log.info("Running damage detection...")

        if self.crop and not self.select_hull:
            self.alg_src = self.det.crop_wheels_circular(self.alg_src)

        if self.write:
            o3d.io.write_point_cloud(self.aligned_path, self.alg_src)
            o3d.io.write_point_cloud(self.tgt_path, self.tgt)

        if self.cc:
            log.info("Running CloudCompare backend")

            o3d.io.write_point_cloud(self.aligned_path, self.alg_src)
            o3d.io.write_point_cloud(self.tgt_path, self.tgt)

            self.ccl.comp_path = self.aligned_path
            self.ccl.ref_path = self.tgt_path

            self.alg_src, self.distances = self.ccl.run_cc(C2C=self.c2c, M3C2=self.m3c2)
            mean, std, threshold = self.det.estimate_noise(
                self.distances,
                percentile=self.percentile,
                sigma_thresh=self.sigma_thresh,
            )
            self.mask = self.distances > threshold

        else:
            self.mask, self.distances, threshold = self.det.detect_damage(
                self.alg_src,
                self.tgt,
                sigma_thresh=self.sigma_thresh,
                percentile=self.percentile,
                bidirectional=True,  # new gate
                outliers_points=self.damage_outliers,
                radius=self.damage_radius,
            )

        log.info("Damage classified above a threshold of %f", threshold)

        if self.crop:
            self.mask = self.det.crop_damage(self.alg_src, self.mask, 55, 10)

        log.info("Damage points: %d / %d", self.mask.sum(), len(self.mask))

        if self.visualise:
            self.det.visualise_colourmap(
                self.alg_src, self.distances, downsample=0.001, write=self.write
            )
            self.det.visualise_binary(
                self.alg_src, self.mask, downsample=0.001, write=self.write
            )

    def _cluster(self):
        log.info(f"Clustering damage regions (Fast cluster: {self.fast_cluster})...")
        if not self.fast_cluster:
            self.labels = self.det.cluster(
                self.alg_src,
                self.mask,
                eps=self.cluster_eps,
                min_samples=self.cluster_min_samples,
                verbose=False,
            )
        else:
            self.labels = self.det.cluster_fast(
                self.alg_src,
                self.mask,
                voxel_size=self.reg.voxel,
                eps=self.cluster_eps,
                min_samples=self.cluster_min_samples,
            )
        self.n_clusters = len(set(self.labels[self.labels >= 0]))
        log.info("Found %d damage cluster(s)", self.n_clusters)

    def _compute_metrics(self):
        log.info("Computing damage metrics...")
        self.metrics = self.det.calculate_damage_metrics(
            self.alg_src, self.distances, self.labels, write_to_pd=True, verbose=False
        )
        self.fp, self.fn = self.det.compare_cluster_runs(
            self.gt_parquet_path, self.guess_parquet_path, 5, True, True
        )

        if self.visualise:
            self.det.color_point_cloud_by_labels(
                self.alg_src, self.labels, downsample=0.001, write=self.write
            )

    def _benchmark(self):
        log.info("Benchmarking registration methods")
        self.reg.benchmark(self.src, self.tgt)


# src = "../data/block/block_damage_accel.ply"
# tgt = "../data/block/block_angle.ply"

# cluster_eps = 0.55
# cluster_samples = 70

# src = "../data/bus/bus_damagev2.ply"
# tgt = "../data/bus/bus.ply"

# src = "../data/bus/bus_7damage.ply"
# tgt = "../data/bus/bus_4damage.ply"


# src = "../data/bus/bus_damagev3.ply"
# tgt = "../data/bus/bus_v2.ply"

# cluster_eps = 1.5
# cluster_samples = 150


# Bus
# pip = Pipeline(
#     src,
#     tgt,
#     plane_fit_dist_th=None,
#     select_hull=False,
#     sor_neighbours=100,
#     sor_std=1.2,
#     voxel_size=5,
#     sigma_thresh=4.0,
#     percentile=80.0,
#     crop=True,
#     cluster_eps=cluster_eps,
#     cluster_min_samples=cluster_samples,
#     fast_cluster=False,
#     min_fitness=0.98,
#     visualise=True,
#     benchmark=False,
#     cc=False,
#     c2c=False,
#     m3c2=True,
#     skip_reg=False,
#     write=False,
# )

# pip.run()

import copy
import logging
import yaml
import numpy as np
import open3d as o3d
from registration import Registration
from damage_detection import DamageDetector
from cloud_compare import CloudCompare

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


class Pipeline:
    def __init__(
        self,
        source_path: str,
        target_path: str,
        plane_fit_dist_th: float = None,
        sor_neighbours: int = None,
        sor_std: float = 1.0,
        voxel_size: float = 2.0,
        sigma_thresh: float = 3.0,
        percentile: float = 80.0,
        median_filter_kernel=None,
        crop: bool = True,
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
        self.plane_fit_dist_th = plane_fit_dist_th

        self.sigma_thresh = sigma_thresh
        self.percentile = percentile
        self.median_filter_kernel = median_filter_kernel
        self.crop = crop
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

        self.src = self.reg.load_pcd(source_path).transform(self.reg.tf)
        self.tgt = self.reg.load_pcd(target_path).transform(self.reg.tf)

        if self.plane_fit_dist_th is not None:
            self.src, removed = self.det.extract_dominant_plane(
                self.src, distance_threshold=plane_fit_dist_th
            )
            self.tgt, _ = self.det.extract_dominant_plane(
                self.tgt, distance_threshold=plane_fit_dist_th
            )
            log.info(
                f"Performed plane segmentation using RANSAC, removed {removed} points"
            )

        if self.sor_neighbours is not None:
            self.src, removed = self.reg.SOR(
                self.src, self.sor_neighbours, self.sor_std
            )
            self.tgt, _ = self.reg.SOR(self.tgt, self.sor_neighbours, self.sor_std)
            log.info(f"Performed Statistical Outlier Removal, removed {removed} points")

        # Results populated by run()
        self.alg_src = None
        self.cropped_pcd = None
        self.transformation = None
        self.mask = None
        self.distances = None
        self.labels = None
        self.metrics = None

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
            self._detect()
            self._cluster()
            self._compute_metrics()
        return self.metrics

    def _register(self):
        log.info("Starting registration...")
        icp, _, eval = self.reg.register(self.src, self.tgt)
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

        if self.crop:
            self.alg_src = self.det.crop_wheels_circular(self.alg_src)

        if self.write:
            o3d.io.write_point_cloud(self.aligned_path, self.alg_src)

        if self.cc:
            log.info("Running CloudCompare backend")

            o3d.io.write_point_cloud(self.aligned_path, self.alg_src)
            o3d.io.write_point_cloud(self.tgt_path, self.tgt)

            self.ccl.comp_path = self.aligned_path
            self.ccl.ref_path = self.tgt_path

            _, self.distances = self.ccl.run_cc(C2C=self.c2c, M3C2=self.m3c2)
            mean, std, threshold = self.det.estimate_noise(
                self.distances,
                percentile=self.percentile,
                sigma_thresh=self.sigma_thresh,
            )
            self.mask = self.distances > threshold

        else:
            # self.mask, self.distances = self.det.detect(
            #     self.alg_src,
            #     self.tgt,
            #     sigma_thresh=self.sigma_thresh,
            #     percentile=self.percentile,
            #     median_filter_kernel=self.median_filter_kernel,
            # )
            self.mask, self.distances = self.det.detect_damage(
                self.alg_src,
                self.tgt,
                sigma_thresh=self.sigma_thresh,
                percentile=self.percentile,
                bidirectional=True,  # new gate
                remove_outliers=False,
            )

        if self.crop:
            self.mask = self.det.crop_damage(self.alg_src, self.mask, 55, 40)

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
            )
        else:
            self.labels = self.det.cluster_fast(
                self.alg_src,
                self.mask,
                voxel_size=self.reg.voxel,
                eps=self.cluster_eps,
                min_samples=self.cluster_min_samples,
            )
        n_clusters = len(set(self.labels[self.labels >= 0]))
        log.info("Found %d damage cluster(s)", n_clusters)

        if self.visualise:
            self.det.color_point_cloud_by_labels(
                self.alg_src, self.labels, downsample=0.001, write=self.write
            )

    def _compute_metrics(self):
        log.info("Computing damage metrics...")
        self.metrics = self.det.calculate_damage_metrics(
            self.alg_src, self.distances, self.labels, grid_res=0.1
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

src = "../data/bus/bus_damagev3.ply"
tgt = "../data/bus/bus_v2.ply"

cluster_eps = 1.5
cluster_samples = 150


# Bus
pip = Pipeline(
    src,
    tgt,
    plane_fit_dist_th=None,
    sor_neighbours=100,
    sor_std=1.2,
    voxel_size=5,
    sigma_thresh=4.0,
    percentile=80.0,
    median_filter_kernel=None,
    crop=True,
    cluster_eps=cluster_eps,
    cluster_min_samples=cluster_samples,
    fast_cluster=False,
    min_fitness=0.825,
    visualise=True,
    benchmark=False,
    cc=False,
    c2c=False,
    m3c2=True,
    skip_reg=False,
    write=False,
)

pip.run()

import copy
import logging
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
        voxel_size: float = 2.0,
        sigma_thresh: float = 3.0,
        percentile: float = 80.0,
        cluster_eps: float = 2.0,
        cluster_min_samples: int = 10,
        min_fitness: float = 0.5,
        visualise: bool = True,
        benchmark=False,
        cc=False,
        c2c=True,
        m3c2=False,
        aligned_path="../data/CC/alg_source_CC.ply",
    ):
        self.reg = Registration(voxel_size)
        self.det = DamageDetector()
        self.ccl = CloudCompare(comp_path=source_path, ref_path=target_path)

        self.sigma_thresh = sigma_thresh
        self.percentile = percentile
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.min_fitness = min_fitness
        self.visualise = visualise
        self.benchmark = benchmark
        self.cc = cc
        self.c2c = c2c
        self.m3c2 = m3c2

        self.aligned_path = aligned_path

        self.src = self.reg.load_pcd(source_path)
        self.tgt = self.reg.load_pcd(target_path)

        # Results populated by run()
        self.alg_src = None
        self.transformation = None
        self.mask = None
        self.distances = None
        self.labels = None
        self.metrics = None

    def run(self):
        if self.benchmark:
            self._benchmark()

        self._register()
        self._detect()
        self._cluster()
        self._compute_metrics()
        return self.metrics

    def _register(self):
        log.info("Starting registration...")
        icp, _ = self.reg.register(self.src, self.tgt)
        log.info("ICP fitness: %.4f  RMSE: %.6f", icp.fitness, icp.inlier_rmse)

        if icp.fitness < self.min_fitness:
            raise RuntimeError(
                f"Registration fitness {icp.fitness:.3f} is below threshold "
                f"{self.min_fitness}. Check inputs or voxel size."
            )

        self.transformation = icp.transformation
        self.alg_src = copy.deepcopy(self.src).transform(self.transformation)

        if self.visualise:
            self.reg.visualise_result(self.src, self.tgt, self.transformation)

    def _detect(self):
        log.info("Running damage detection...")

        if self.cc:
            log.info("Running CloudCompare backend")
            o3d.io.write_point_cloud(self.aligned_path, self.alg_src)
            _, self.distances = self.ccl.run_cc(C2C=self.c2c, M3C2=self.m3c2)
            mean, std, threshold = self.det.estimate_noise(
                self.distances,
                percentile=self.percentile,
                sigma_thresh=self.sigma_thresh,
            )
            self.mask = self.distances > threshold

        else:
            self.mask, self.distances = self.det.detect(
                self.alg_src,
                self.tgt,
                sigma_thresh=self.sigma_thresh,
                percentile=self.percentile,
            )

        log.info("Damage points: %d / %d", self.mask.sum(), len(self.mask))

        if self.visualise:
            self.det.visualise_colourmap(self.alg_src, self.distances)
            self.det.visualise_binary(self.alg_src, self.mask)

    def _cluster(self):
        log.info("Clustering damage regions...")
        self.labels = self.det.cluster(
            self.alg_src,
            self.mask,
            eps=self.cluster_eps,
            min_samples=self.cluster_min_samples,
        )
        n_clusters = len(set(self.labels[self.labels >= 0]))
        log.info("Found %d damage cluster(s)", n_clusters)

        if self.visualise:
            self.det.color_point_cloud_by_labels(self.alg_src, self.labels)

    def _compute_metrics(self):
        log.info("Computing damage metrics...")
        self.metrics = self.det.calculate_damage_metrics(
            self.alg_src, self.distances, self.labels
        )

    def _benchmark(self):
        log.info("Benchmarking registration methods")
        self.reg.benchmark(self.src, self.tgt)


src = "../data/CC/SRC.ply"
tgt = "../data/CC/TGT.ply"

# src = "../data/pcd/Reg_block_tripledented_random.ply"
# tgt = "../data/pcd/Reg_block_2_random.ply"

pip = Pipeline(
    src,
    tgt,
    voxel_size=3.0,
    sigma_thresh=3.0,
    percentile=90.0,
    cluster_eps=2.0,
    cluster_min_samples=20,
    min_fitness=0.5,
    visualise=True,
    benchmark=False,
    cc=False,
    c2c=True,
    m3c2=False,
    aligned_path="../data/CC/alg_source_CC.ply",
)
pip.run()

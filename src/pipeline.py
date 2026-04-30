import copy
import logging
import open3d as o3d
from registration import Registration
from damage_detection import DamageDetector

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
    ):
        self.reg = Registration(voxel_size)
        self.det = DamageDetector()

        self.sigma_thresh = sigma_thresh
        self.percentile = percentile
        self.cluster_eps = cluster_eps
        self.cluster_min_samples = cluster_min_samples
        self.min_fitness = min_fitness
        self.visualise = visualise
        self.benchmark = benchmark

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

    def _detect(self):
        log.info("Running damage detection...")
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


# src = "../data/pcd/CC/SRC.ply"
# tgt = "../data/pcd/CC/TGT.ply"

src = "../data/pcd/Reg_block_tripledented_random.ply"
tgt = "../data/pcd/Reg_block_2_random.ply"

pip = Pipeline(
    src,
    tgt,
    voxel_size=2.0,
    sigma_thresh=3.0,
    percentile=90.0,
    cluster_eps=2.0,
    cluster_min_samples=20,
    min_fitness=0.5,
    visualise=True,
    benchmark=False,
)
pip.run()

import numpy as np
import open3d as o3d

from global_reg import Registration
from damage_detection import DamageDetector

reg = Registration(2)
det = DamageDetector()


class Pipeline:
    def __init__(self, source_path, target_path):
        self.src = reg.load_pcd(source_path)
        self.tgt = reg.load_pcd(target_path)
        self.alg_src = o3d.geometry.PointCloud()

    def run(self):
        # reg.set_voxel(self.src, ratio=0.02)

        # reg.benchmark(self.src, self.tgt)

        icp, _ = reg.register(self.src, self.tgt)
        tf = icp.transformation

        self.alg_src = self.src.transform(tf)
        self.alg_src, self.tgt = reg.downsample(self.alg_src, 0.01), reg.downsample(
            self.tgt, 0.01
        )

        mask, distances, _ = det.detect(self.alg_src, self.tgt, sigma_thresh=3)

        det.visualise_colourmap(self.alg_src, distances=distances)

        det.visualise_binary(self.alg_src, mask)

        labels = det.cluster(
            aligned_source=self.alg_src, damage_mask=mask, eps=2, min_samples=20
        )

        dict = det.calculate_damage_metrics(self.alg_src, distances, labels)

        det.color_point_cloud_by_labels(self.alg_src, labels)


src = "../data/Reg_block_tripledented_abrupt_50000.ply"
tgt = "../data/Reg_block_2_smooth.ply"

# src = "../data/CC/SRC.ply"
# tgt = "../data/CC/TGT.ply"

pip = Pipeline(src, tgt)
pip.run()

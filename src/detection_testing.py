import copy
import matplotlib.pyplot as plt

from damage_detection import DamageDetector
from global_reg import Registration

reg = Registration(course_voxel=3, voxel_size=3)
det = DamageDetector()

f1 = "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block.stl"
f2 = "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block_tripledented.stl"

tgt, src = reg.poisson_convert(f1), reg.poisson_convert(f2)

icp, _ = reg.register(src, tgt)

aligned_src = copy.deepcopy(src)
aligned_src.transform(icp.transformation)

distance, _ = det.compute_bidirectional_c2c(aligned_source=aligned_src, target=tgt)

mean, std, _ = det.estimate_noise(distance, 80)

mask, distances, _ = det.detect(
    aligned_source=aligned_src, target=tgt, noise_floor=mean, noise_std=std
)

labels = det.cluster(aligned_source=aligned_src, damage_mask=mask)

det.color_point_cloud_by_labels(aligned_src, labels)

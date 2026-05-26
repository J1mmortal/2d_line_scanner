from registration import Registration
from damage_detection import DamageDetector
from cloud_compare import CloudCompare
import open3d as o3d
import numpy as np

from scipy.spatial import ConvexHull, cKDTree

reg = Registration(5)
det = DamageDetector()

# tgt_p = "../data/bus/bus_damagev3.ply"
# tgt_p = "../data/bus/bus_v2.ply"

tgt_p = "../data/bus/bus3.ply"
# tgt_p = "../data/bus/damage3.ply"
# tgt_p = "../data/bus/damage_80fps.ply"

src_p = "../data/bus/damage3.ply"

tgt = reg.load_pcd(tgt_p)
src = reg.load_pcd(src_p)

src = reg.downsample(src, ratio=0.00075)
tgt = reg.downsample(tgt, ratio=0.00075)

reg.set_voxel(src, ratio=0.01)
print(reg.voxel)

reg.benchmark(src, tgt)
# tgt, _ = reg.SOR(tgt, 60, 4)

# det.select_bus_hull(tgt, eps=2.0)

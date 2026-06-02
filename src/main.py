import logging

logging.basicConfig(
    # filename="../data/run.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler("../data/run.log", mode="w"),
        logging.StreamHandler(),
    ],
    force=True,
)

from pipeline import Pipeline

# Point clouds
# src = "../data/bus/bus_damagev3.ply"
# tgt = "../data/bus/bus_v2.ply"

# src3 = "../data/bus/damage3.ply"
# tgt3 = "../data/bus/bus3.ply"

# src = "../data/bus/damage_80fps.ply"
# tgt = "../data/bus/bus_80fps.ply"

# src_test = "../data/bus/bus4_damage.ply"
# tgt_test = "../data/bus/gt/bus4.ply"

tgt_test = "../data/bus/gt/bus4.ply"
src_test = "../data/bus/lighting/bus4_lightson.ply"

# Initial plane / hull fitting / cropping
plane_fit_dist_th = None
select_hull = True

# Statistical outlier removal parameters (commented values work perfectly without hull, work perfectly with hull) for src3, tgt3
sor_neighbours = None  # 100, 80
sor_std = 4  # 1.2, 4

# Registsration parameters
voxel_size = 5
min_fitness = 0.98

# Noise estimation parameters
sigma_thresh = 4.0  # 4.0
percentile = 80  # 80

# Clustering parameters
cluster_eps = 1.4  # 1.5, 1.1
cluster_samples = 300  # 150, 210
fast_cluster = False

# CloudCompare parameters
cc = False
c2c = False
m3c2 = True

# Flags
visualise = True
benchmark = False
skip_reg = False
write = True
crop = True

pip = Pipeline(
    src_test,
    tgt_test,
    plane_fit_dist_th=plane_fit_dist_th,
    select_hull=select_hull,
    sor_neighbours=sor_neighbours,
    sor_std=sor_std,
    voxel_size=voxel_size,
    sigma_thresh=sigma_thresh,
    percentile=percentile,
    crop=crop,
    cluster_eps=cluster_eps,
    cluster_min_samples=cluster_samples,
    fast_cluster=fast_cluster,
    min_fitness=min_fitness,
    visualise=visualise,
    benchmark=benchmark,
    cc=cc,
    c2c=c2c,
    m3c2=m3c2,
    skip_reg=skip_reg,
    write=write,
)

pip.run()

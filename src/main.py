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

### --- Points clouds tested ---

# src_test = "../data/bus/bus4_damage.ply"
tgt_test = "../data/bus/gt/bus4.ply"
src_test = "../data/bus/snelheid_test_bus4.ply"

# tgt_sticker = "../data/bus/stickers/bus1_stpr.ply"
# src_sticker = "../data/bus/stickers/bus1_sticker.ply"

# tgt_dry = "../data/bus/wet/bus3_dry.ply"
# src_wet = "../data/bus/wet/bus3_wet.ply"
# src_superwet = "../data/bus/wet/bus3_superwet.ply"

# src_sun = r"..\data\bus\lighting\bus4_sun.ply"
# src_light = r"..\data\bus\lighting\bus4_lightson.ply"
# src_dark = r"..\data\bus\lighting\bus4_lightsoff.ply"

# tgt_sun = r"..\data\bus\lighting\bus3_sun.ply"
# tgt_dark = r"..\data\bus\lighting\bus3_lightsoff.ply"

# tgt_0 = "../data/bus/angles/bus3_0.ply"
# src_05 = "../data/bus/angles/bus3_0.5.ply"
# src_15 = "../data/bus/angles/bus3_1.5.ply"
# src_07 = "../data/bus/angles/bus3_-0.7.ply"
# src_min15 = "../data/bus/angles/bus3_-1.5.ply"
# src_min20 = "../data/bus/angles/bus3_-2.0.ply"
# src_0 = "../data/bus/angles/bus4_0.ply"
# src_1 = "../data/bus/angles/bus4_1.ply"
# src_2 = "../data/bus/angles/bus4_2.ply"
# src_min1 = "../data/bus/angles/bus4_-1.ply"
# src_min2 = "../data/bus/angles/bus4_-2.ply"


# Initial plane / hull fitting / cropping
select_hull = True
velocity_scale = True

# Statistical outlier removal parameters (commented values work perfectly without hull, work perfectly with hull) for src3, tgt3
sor_neighbours = None  # 100, 80
sor_std = 4  # 1.2, 4

# Registsration parameters
voxel_size = 5
min_fitness = 0.95
downsample_reg = False

# Noise estimation parameters (4.0, 80)
sigma_thresh = 3.0  # 4.0
percentile = 80  # 80

# Remove outliers from damage mask
damage_points = 200
damage_radius = 2

# Clustering parameters (1.4, 300)
cluster_eps = 1.1  # 1.5, 1.1
cluster_samples = 250  # 150, 210
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
    select_hull=select_hull,
    velocity_scale=velocity_scale,
    sor_neighbours=sor_neighbours,
    sor_std=sor_std,
    voxel_size=voxel_size,
    downsample_reg=downsample_reg,
    sigma_thresh=sigma_thresh,
    percentile=percentile,
    crop=crop,
    damage_outliers=damage_points,
    damage_radius=damage_radius,
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

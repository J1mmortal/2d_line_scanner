from pipeline import Pipeline

# Point clouds
src = "../data/bus/bus_damagev3.ply"
tgt = "../data/bus/bus_v2.ply"

# Initial plane / hull fitting / cropping
plane_fit_dist_th = None
select_hull = True

# Statistical outlier removal parameters (commented values work perfectly without hull)
sor_neighbours = 80  # 100
sor_std = 4  # 1.2

# Registsration parameters
voxel_size = 5
min_fitness = 0.98

# Noise estimation parameters
sigma_thresh = 4.0
percentile = 80.0

# Clustering parameters
cluster_eps = 1.1  # 1.5
cluster_samples = 210  # 150
fast_cluster = False

# CloudCompare parameters
cc = False
c2c = False
m3c2 = True

# Flags
visualise = True
benchmark = False
skip_reg = False
write = False
crop = True

pip = Pipeline(
    src,
    tgt,
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

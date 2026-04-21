import open3d as o3d
from global_reg import Registration
import copy

dataset = o3d.data.DemoICPPointClouds()
src = o3d.io.read_point_cloud(dataset.paths[0])
tgt = o3d.io.read_point_cloud(dataset.paths[1])

# # dataset = o3d.data.DemoColoredICPPointClouds()
# # src = o3d.io.read_point_cloud(dataset.paths[0])
# # tgt = o3d.io.read_point_cloud(dataset.paths[1])

reg = Registration()
icp_result, global_result = reg.register(src, tgt)

tf = icp_result.transformation
print(icp_result.inlier_rmse, global_result.inlier_rmse)

src.paint_uniform_color([1, 0.7, 0])
tgt.paint_uniform_color([0, 0.65, 1])

alg_src = copy.deepcopy(src)
alg_src.transform(tf)

o3d.visualization.draw_geometries(
    [src, tgt], window_name="BEFORE", width=800, height=600
)

o3d.visualization.draw_geometries(
    [alg_src, tgt], window_name="AFTER", width=800, height=600
)

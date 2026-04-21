import open3d as o3d
import numpy as np
import copy

from global_reg import Registration

# reg = Registration()


mesh1 = o3d.io.read_triangle_mesh(
    "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block.stl"
)
mesh1.compute_vertex_normals()

tgt = mesh1.sample_points_poisson_disk(number_of_points=10_000)
tgt.paint_uniform_color([0, 0.65, 1])

mesh2 = o3d.io.read_triangle_mesh(
    "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block_2.stl"
)
mesh2.compute_vertex_normals()

src = mesh2.sample_points_poisson_disk(number_of_points=10_000)
src.paint_uniform_color([1, 0.7, 0])


# o3d.visualization.draw_geometries([tgt, src], window_name="Block")

reg = Registration()
icp_result, global_result = reg.register(src, tgt)

tf = icp_result.transformation
print(icp_result.inlier_rmse, global_result.inlier_rmse)

src.paint_uniform_color([1, 0.7, 0])  # orange
tgt.paint_uniform_color([0, 0.65, 1])  # blue

alg_src = copy.deepcopy(src)
alg_src.transform(tf)

# Show before and after in two windows
o3d.visualization.draw_geometries(
    [src, tgt], window_name="BEFORE", width=800, height=600
)

o3d.visualization.draw_geometries(
    [alg_src, tgt], window_name="AFTER", width=800, height=600
)

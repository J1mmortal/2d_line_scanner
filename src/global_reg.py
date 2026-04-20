import open3d as o3d
import numpy as np
import copy


class Registration:
    def __init__(self):
        self.voxel_size = 0.05  # cm
        self.course_voxel = 0.05

        self.pcd = o3d.geometry.PointCloud()

    def convert_file(self, file):
        if file is not None:
            mesh = o3d.io.read_triangle_mesh(file)
            mesh.compute_vertex_normals()

            self.pcd.points = mesh.vertices
            self.pcd.normals = mesh.vertex_normals

    def preprocess(self, pcd):
        pcd_down = pcd.voxel_down_sample(self.course_voxel)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.course_voxel * 2, max_nn=30
            )
        )
        return pcd_down

    def compute_fpfh(self, pcd):
        if not pcd.has_normals():
            raise RuntimeError("Compute normals before FPFH")

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.voxel_size * 5, max_nn=100
            ),
        )
        return fpfh

    def global_registration_ransac(self, source, target, source_fpfh, target_fpfh):

        distance_threshold = self.course_voxel * 1.5

        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,  # only keep symmetric matches
            max_correspondence_distance=distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
            ransac_n=3,  # triplets of correspondences
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    distance_threshold
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=100_000, confidence=0.999
            ),
        )
        return result

    def refine_icp(self, source, target, init_transform):
        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=self.voxel_size * 0.4,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        )
        return result

    def register(self, source, target):
        src_down = self.preprocess(source)
        tgt_down = self.preprocess(target)

        src_fpfh = self.compute_fpfh(src_down)
        tgt_fpfh = self.compute_fpfh(tgt_down)

        ransac_result = self.global_registration_ransac(
            src_down, tgt_down, src_fpfh, tgt_fpfh
        )

        # transformation = ransac_result.transformation
        # fitness = ransac_result.fitness
        # rmse = ransac_result.inlier_rmse

        icp_result = self.refine_icp(source, target, ransac_result.transformation)

        return icp_result, ransac_result


dataset = o3d.data.DemoICPPointClouds()
src = o3d.io.read_point_cloud(dataset.paths[0])
tgt = o3d.io.read_point_cloud(dataset.paths[1])

# dataset = o3d.data.DemoColoredICPPointClouds()
# src = o3d.io.read_point_cloud(dataset.paths[0])
# tgt = o3d.io.read_point_cloud(dataset.paths[1])

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

# vis = o3d.visualization.Visualizer()
# vis.create_window(window_name="Registration Result", width=1280, height=720)

# vis.add_geometry(alg_src)
# vis.add_geometry(tgt)

# # Optional: render options
# opt = vis.get_render_option()
# opt.point_size = 2.0
# opt.background_color = [0.1, 0.1, 0.1]  # dark background

# vis.run()  # blocks until you close the window
# vis.destroy_window()

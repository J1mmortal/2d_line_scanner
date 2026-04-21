import open3d as o3d
import numpy as np
import copy


class Registration:
    def __init__(self):
        # Main settings shared by global and local registration
        self.voxel_size = 0.05
        self.coarse_voxel = 0.05
        self.max_correspondence_distance = self.voxel_size * 0.4

        self.relative_fitness = 1e-6
        self.relative_rmse = 1e-6
        self.max_iteration = 50

        self.normal_radius = self.voxel_size * 2
        self.normal_max_nn = 30
        self.fpfh_radius = self.voxel_size * 5
        self.fpfh_max_nn = 100
        self.ransac_distance_threshold = self.coarse_voxel * 1.5
        self.ransac_max_iteration = 100_000
        self.ransac_confidence = 0.999

        self.pcd = o3d.geometry.PointCloud()

    def get_icp_criteria(self):
        return o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=self.relative_fitness,
            relative_rmse=self.relative_rmse,
            max_iteration=self.max_iteration,
        )

    def ensure_normals(self, pcd):
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.normal_radius,
                    max_nn=self.normal_max_nn,
                )
            )
        return pcd

    def convert_file(self, file):
        if file is not None:
            mesh = o3d.io.read_triangle_mesh(file)
            mesh.compute_vertex_normals()

            self.pcd.points = mesh.vertices
            self.pcd.normals = mesh.vertex_normals

    def preprocess(self, pcd):
        pcd_down = pcd.voxel_down_sample(self.coarse_voxel)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn,
            )
        )
        return pcd_down

    def compute_fpfh(self, pcd):
        if not pcd.has_normals():
            raise RuntimeError("Compute normals before FPFH")

        fpfh = o3d.pipelines.registration.compute_fpfh_feature(
            pcd,
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.fpfh_radius,
                max_nn=self.fpfh_max_nn,
            ),
        )
        return fpfh

    def global_registration_ransac(self, source, target, source_fpfh, target_fpfh):
        result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
            source,
            target,
            source_fpfh,
            target_fpfh,
            mutual_filter=True,
            max_correspondence_distance=self.ransac_distance_threshold,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
            ransac_n=3,
            checkers=[
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                    self.ransac_distance_threshold
                ),
            ],
            criteria=o3d.pipelines.registration.RANSACConvergenceCriteria(
                max_iteration=self.ransac_max_iteration,
                confidence=self.ransac_confidence,
            ),
        )
        return result

    def refine_icp(self, source, target, init_transform):
        source = self.ensure_normals(copy.deepcopy(source))
        target = self.ensure_normals(copy.deepcopy(target))

        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=self.max_correspondence_distance,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(),
            criteria=self.get_icp_criteria(),
        )
        return result

    def gen_icp(self, source, target, init_transform):
        source = self.ensure_normals(copy.deepcopy(source))
        target = self.ensure_normals(copy.deepcopy(target))

        result = o3d.pipelines.registration.registration_generalized_icp(
            source,
            target,
            max_correspondence_distance=self.max_correspondence_distance,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria=self.get_icp_criteria(),
        )
        return result

    def multi_scale_icp(self, source, target, init_transform):
        voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
        max_iter = o3d.utility.IntVector([50, 30, 14])

        criteria_list = [
            o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=it)
            for it in max_iter
        ]

        result = o3d.t.pipelines.registration.multi_scale_icp(
            source,
            target,
            voxel_sizes,
            criteria_list,
            max_correspondence_distances=o3d.utility.DoubleVector([0.3, 0.15, 0.075]),
            init_source_to_target=init_transform,
            estimation=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
        )

        return result

    def get_initial_guess(self, source, target):
        src_down = self.preprocess(source)
        tgt_down = self.preprocess(target)

        src_fpfh = self.compute_fpfh(src_down)
        tgt_fpfh = self.compute_fpfh(tgt_down)

        ransac_result = self.global_registration_ransac(
            src_down, tgt_down, src_fpfh, tgt_fpfh
        )

        return ransac_result

    def register(self, source, target):
        ransac_result = self.get_initial_guess(source, target)
        icp_result = self.refine_icp(source, target, ransac_result.transformation)
        return icp_result, ransac_result


dataset = o3d.data.DemoICPPointClouds()
src = o3d.io.read_point_cloud(dataset.paths[0])
tgt = o3d.io.read_point_cloud(dataset.paths[1])

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
import open3d as o3d
import numpy as np
import copy
import time


class Registration:
    def __init__(self, voxel_size=0.05, use_gpu = False):

        self.voxel = voxel_size
        self.use_gpu = self

        # ICP correspondance distances
        self.max_correspondence_distance = self.voxel * 0.6

        # For estimating normals
        self.normal_radius = self.voxel * 2
        self.normal_max_nn = 30

        # For computing FPFH features
        self.fpfh_radius = self.voxel * 5
        self.fpfh_max_nn = 100

        # RANSAC parameters
        self.ransac_distance_threshold = self.voxel * 1.5
        self.ransac_max_iteration = 100_000
        self.ransac_confidence = 0.999

        self.pcd = o3d.geometry.PointCloud()

        # Convergence criteria
        self.relative_fitness = 1e-6
        self.relative_rmse = 1e-6
        self.max_iteration = 150

        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=self.relative_fitness,
            relative_rmse=self.relative_rmse,
            max_iteration=self.max_iteration,
        )

    def load_pcd(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd

    def ensure_normals(self, pcd):
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.normal_radius,
                    max_nn=self.normal_max_nn,
                )
            )
        return pcd

    def simple_convert(self, file_path):
        if file_path is not None:
            mesh = o3d.io.read_triangle_mesh(file_path)
            mesh.compute_vertex_normals()

            self.pcd.points = mesh.vertices
            self.pcd.normals = mesh.vertex_normals
        return self.pcd

    def poisson_convert(self, file, n_points=50000):
        mesh = o3d.io.read_triangle_mesh(file)
        mesh.compute_vertex_normals()
        pcd = mesh.sample_points_poisson_disk(number_of_points=n_points)

        return pcd

    def set_voxel(self, pcd, ratio=0.02):
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_dimension = np.max(extent)

        # Set voxel size to X% of the largest dimension (e.g., 0.01 = 1%)
        self.voxel = max_dimension * ratio
        print(f"Max dimension: {max_dimension}; Voxel size: {self.voxel}")

    def preprocess(self, pcd):
        pcd_down = pcd.voxel_down_sample(self.voxel)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn,
            )
        )
        return pcd_down

    def SOR(self, pcd, neigbours, std_ratio):
        filtered_cloud, ind = pcd.remove_statistical_outlier(
            nb_neighbors=neigbours, std_ratio=std_ratio
        )
        removed = len(pcd.points) - len(filtered_cloud.points)

        return filtered_cloud, removed

    def radius_outlier_removal(self, pcd, n_points, radius):
        filtered_cloud, ind = pcd.remove_radius_outlier(
            nb_points=n_points, radius=radius
        )
        removed = len(pcd.points) - len(filtered_cloud.points)

        return filtered_cloud, removed

    def downsample(self, pcd, ratio):
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_dimension = np.max(extent)

        # Set voxel size to X% of the largest dimension (e.g., 0.01 = 1%)
        dynamic_voxel = max_dimension * ratio

        # Downsample
        pcd_down = pcd.voxel_down_sample(voxel_size=dynamic_voxel)

        # Estimate normals. radius must scale with the dynamic voxel.
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=dynamic_voxel * 2.5,
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
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(
                False
            ),
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

    def icp(self, source, target, init_transform):
        source = self.ensure_normals(source)
        target = self.ensure_normals(target)

        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=self.max_correspondence_distance,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            criteria=self.criteria,
        )
        return result

    def plane_icp(self, source, target, init_transform, K=10):
        source = self.ensure_normals(source)
        target = self.ensure_normals(target)

        # Downweights points with large residuals. Should reduce effect of damaged region on registration
        tukey_kernel = o3d.pipelines.registration.TukeyLoss(k=K)

        result = o3d.pipelines.registration.registration_icp(
            source,
            target,
            max_correspondence_distance=self.max_correspondence_distance,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationPointToPlane(
                tukey_kernel
            ),
            criteria=self.criteria,
        )
        return result

    def gen_icp(self, source, target, init_transform):
        source = self.ensure_normals(source)
        target = self.ensure_normals(target)

        result = o3d.pipelines.registration.registration_generalized_icp(
            source,
            target,
            max_correspondence_distance=self.max_correspondence_distance,
            init=init_transform,
            estimation_method=o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria=self.criteria,
        )
        return result

    # def multi_scale_icp(self, source, target, init_transform):
    #     voxel_sizes = o3d.utility.DoubleVector([0.1, 0.05, 0.025])
    #     max_iter = o3d.utility.IntVector([50, 30, 14])

    #     criteria_list = [
    #         o3d.t.pipelines.registration.ICPConvergenceCriteria(max_iteration=it)
    #         for it in max_iter
    #     ]

    #     result = o3d.t.pipelines.registration.multi_scale_icp(
    #         source,
    #         target,
    #         voxel_sizes,
    #         criteria_list,
    #         max_correspondence_distances=o3d.utility.DoubleVector([0.3, 0.15, 0.075]),
    #         init_source_to_target=init_transform,
    #         estimation=o3d.t.pipelines.registration.TransformationEstimationPointToPlane(),
    #     )

    #     return result

    def get_initial_guess(self, source, target):
        src_down = self.preprocess(source)
        tgt_down = self.preprocess(target)

        src_fpfh = self.compute_fpfh(src_down)
        tgt_fpfh = self.compute_fpfh(tgt_down)

        ransac_result = self.global_registration_ransac(
            src_down, tgt_down, src_fpfh, tgt_fpfh
        )

        return ransac_result

    def register(self, source, target, use_gpu=False):
        ransac_result = self.get_initial_guess(source, target)

        # if use_gpu:
        #     icp_result = self.gpu_icp(source, target, ransac_result.transformation)
        # else:
        icp_result = self.gen_icp(source, target, ransac_result.transformation)

        return icp_result, ransac_result

    def evaluate_alignment(self, source, target, transform):
        return o3d.pipelines.registration.evaluate_registration(
            source,
            target,
            transformation=transform,
            max_correspondence_distance=self.max_correspondence_distance,
        )

    def visualise_result(
        self, source, target=None, transform=np.eye(4), downsample=0.008
    ):  # Downsample sets voxel size: =1 gives one voxel for the whole point cloud
        if target is not None:
            src_d = self.downsample(source, ratio=downsample)
            tgt_d = self.downsample(target, ratio=downsample)

            src_d.paint_uniform_color([1, 0.2, 0])
            tgt_d.paint_uniform_color([0, 0.65, 0.93])

            src_d.transform(transform)

            title = f"Alignment after transformation with {transform}"

            o3d.visualization.draw_geometries(
                [src_d, tgt_d], window_name=title, width=1000, height=800
            )
        else:
            src_d = self.downsample(source, ratio=downsample)

            src_d.paint_uniform_color([1, 0.2, 0])

            src_d.transform(transform)

            title = f"Point cloud visualisation {transform}"

            o3d.visualization.draw_geometries(
                [src_d], window_name=title, width=1000, height=800
            )

    def rank_results(self, results):
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]

        # Sort by highest fitness, then lowest RMSE, then lowest runtime
        successful.sort(key=lambda r: (-r["fitness"], r["inlier_rmse"], r["runtime_s"]))
        return successful + failed

    def benchmark_method(self, method_fn, source, target, init_guess):
        start = time.perf_counter()

        try:
            result = method_fn(source, target, init_guess)
            runtime_s = time.perf_counter() - start

            evaluation = self.evaluate_alignment(source, target, result.transformation)

            return {
                "method": f"{method_fn.__name__}",
                "success": True,
                "fitness": evaluation.fitness,
                "inlier_rmse": evaluation.inlier_rmse,
                "runtime_s": runtime_s,
                "transformation": result.transformation,
                "result": result,
                "evaluation": evaluation,
                "threshold": self.max_correspondence_distance,
                "max_iter": self.max_iteration,
                "notes": "",
            }
        except Exception as e:
            runtime_s = time.perf_counter() - start
            return {
                "method": f"{method_fn.__name__}",
                "success": False,
                "fitness": None,
                "inlier_rmse": None,
                "runtime_s": runtime_s,
                "transformation": None,
                "result": None,
                "evaluation": None,
                "threshold": self.max_correspondence_distance,
                "max_iter": self.max_iteration,
                "notes": f"Failed: {str(e)}",
            }

    def benchmark_global_method(self, source, target):
        start = time.perf_counter()

        try:
            # get_initial_guess handles downsampling, FPFH extraction, and RANSAC
            result = self.get_initial_guess(source, target)
            runtime_s = time.perf_counter() - start

            # Evaluate the global alignment against the original dense point clouds
            evaluation = self.evaluate_alignment(source, target, result.transformation)

            return {
                "method": "FPFH + RANSAC",
                "success": True,
                "fitness": evaluation.fitness,
                "inlier_rmse": evaluation.inlier_rmse,
                "runtime_s": runtime_s,
                "transformation": result.transformation,
                "result": result,
                "evaluation": evaluation,
                "threshold": self.ransac_distance_threshold,
                "max_iter": self.ransac_max_iteration,
                "notes": "",
            }
        except Exception as e:
            runtime_s = time.perf_counter() - start
            return {
                "method": "FPFH + RANSAC",
                "success": False,
                "fitness": None,
                "inlier_rmse": None,
                "runtime_s": runtime_s,
                "transformation": None,
                "result": None,
                "evaluation": None,
                "threshold": self.ransac_distance_threshold,
                "max_iter": self.ransac_max_iteration,
                "notes": f"Failed: {str(e)}",
            }

    def print_result_summary(self, results):
        ranked_results = self.rank_results(results)

        print("===== Ranked benchmark summary =====")
        header = (
            f"{'Rank':<6}{'Method':<24}{'Fitness':<14}"
            f"{'Inlier RMSE':<16}{'Runtime (s)':<14}{'Threshold':<12}{'Max iter':<10}"
        )
        print(header)
        print("-" * len(header))

        for idx, item in enumerate(ranked_results, start=1):
            fitness = f"{item['fitness']:.6f}" if item["fitness"] is not None else "N/A"
            rmse = (
                f"{item['inlier_rmse']:.6f}"
                if item["inlier_rmse"] is not None
                else "N/A"
            )
            runtime_s = f"{item['runtime_s']:.6f}"

            print(
                f"{idx:<6}{item['method']:<24}{fitness:<14}"
                f"{rmse:<16}{runtime_s:<14}{item['threshold']:<12.4f}{item['max_iter']:<10}"
            )

            if item["notes"]:
                print(f"      Notes: {item['notes']}")

    def benchmark(self, src, tgt):
        results = []

        # 1. Benchmark RANSAC (Global)
        global_benchmark = self.benchmark_global_method(src, tgt)
        results.append(global_benchmark)

        # Extract the initial guess for the local methods
        init_guess = global_benchmark["transformation"]

        # 2. Benchmark ICP variants (Local)
        if global_benchmark["success"]:
            results.append(self.benchmark_method(self.icp, src, tgt, init_guess))
            results.append(self.benchmark_method(self.plane_icp, src, tgt, init_guess))
            results.append(self.benchmark_method(self.gen_icp, src, tgt, init_guess))

        # 3. Print the unified table
        self.print_result_summary(results)

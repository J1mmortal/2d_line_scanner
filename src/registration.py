import open3d as o3d
import open3d.core as o3c
import numpy as np
import copy
import time
import logging


class Registration:
    def __init__(self, voxel_size=0.05):
        self.voxel = voxel_size
        self.device = o3c.Device("CPU:0")
        self.float_dtype = o3c.float32

        self.tf = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # ICP correspondance distances
        self.max_correspondence_distance = self.voxel * 0.4

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

        # class _LegacyCompatibleResult:
        #     """Helper to cast Tensor API results back to standard NumPy/Legacy formats"""

        #     def __init__(self, tensor_result):
        #         # Transformation must be float64 for o3d legacy transform()
        #         self.transformation = (
        #             tensor_result.transformation.cpu().numpy().astype(np.float64)
        #         )
        #         self.fitness = tensor_result.fitness
        #         self.inlier_rmse = tensor_result.inlier_rmse

    def load_pcd(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd

    def _format_tensor_result(self, tensor_result):
        from types import SimpleNamespace

        return SimpleNamespace(
            transformation=tensor_result.transformation.cpu()
            .numpy()
            .astype(np.float64),
            fitness=tensor_result.fitness,
            inlier_rmse=tensor_result.inlier_rmse,
        )

    def _to_init_tensor(self, transform: np.ndarray) -> "o3d.core.Tensor":
        return o3c.Tensor(transform, dtype=o3c.float64)

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

    def crop_pcd(
        self,
        pcd,
        max_y_threshold=55,
        x_thresh=40,
        height_axis=1,
        width_axis=0,
        robust_floor=True,
    ):
        xyz = np.asarray(pcd.points)
        heights = xyz[:, height_axis]
        widths = xyz[:, width_axis]

        # Calculate floor relative to actual point distribution
        floor_y = np.percentile(heights, 1) if robust_floor else np.min(heights)
        abs_thresh_y = floor_y + max_y_threshold

        floor_x = np.percentile(widths, 1) if robust_floor else np.min(widths)
        roof_x = np.percentile(widths, 99) if robust_floor else np.max(widths)
        min_thresh_x = floor_x + x_thresh
        max_thresh_x = roof_x - x_thresh

        # Create spatial mask and intersect with damage mask
        valid_height_mask = heights <= abs_thresh_y
        valid_width_mask = (min_thresh_x <= widths) & (widths <= max_thresh_x)
        filtered_mask = valid_height_mask & valid_width_mask

        cropped_xyz = xyz[filtered_mask]

        cropped_pcd = o3d.geometry.PointCloud()
        cropped_pcd.points = o3d.utility.Vector3dVector(cropped_xyz)

        removed_count = len(xyz) - filtered_mask.sum()
        y_name = ["X", "Y", "Z"][height_axis]
        x_name = ["X", "Y", "Z"][width_axis]

        logging.info(
            "Height and width filter (Rel %s: %.2fm; %s: %.2fm) removed %d points.",
            y_name,
            max_y_threshold,
            x_name,
            x_thresh,
            removed_count,
        )

        return cropped_pcd

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

        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        return o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.max_correspondence_distance,
            init_transform,
            estimation,
            self.criteria,
        )

    def plane_icp(self, source, target, init_transform, K=10):
        source = self.ensure_normals(source)
        target = self.ensure_normals(target)

        tukey = o3d.pipelines.registration.TukeyLoss(k=K)
        estimation = o3d.pipelines.registration.TransformationEstimationPointToPlane(
            tukey
        )
        return o3d.pipelines.registration.registration_icp(
            source,
            target,
            self.max_correspondence_distance,
            init_transform,
            estimation,
            self.criteria,
        )

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

    def multi_scale_icp(self, source, target, init_transform):
        source = self.ensure_normals(source)
        target = self.ensure_normals(target)

        # Uses whichever device was configured in __init__ (GPU or fallback CPU)
        t_device = getattr(self, "device", o3c.Device("CPU:0"))
        t_dtype = getattr(self, "float_dtype", o3c.float32)

        t_src = o3d.t.geometry.PointCloud.from_legacy(source, t_dtype, t_device)
        t_tgt = o3d.t.geometry.PointCloud.from_legacy(target, t_dtype, t_device)

        estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()

        # Coarse-to-fine parameter configuration
        voxel_sizes = o3d.utility.DoubleVector([self.voxel, self.voxel / 4])
        max_corrs = o3d.utility.DoubleVector(
            [
                self.max_correspondence_distance * 2,
                self.max_correspondence_distance,
            ]
        )

        max_iter = o3d.utility.IntVector([30, self.max_iteration])
        criteria_list = [
            o3d.t.pipelines.registration.ICPConvergenceCriteria(
                self.relative_fitness, self.relative_rmse, it
            )
            for it in max_iter
        ]

        init_tensor = o3d.core.Tensor(init_transform, dtype=o3d.core.float64)
        result = o3d.t.pipelines.registration.multi_scale_icp(
            t_src,
            t_tgt,
            voxel_sizes,
            criteria_list,
            max_corrs,
            init_tensor,
            estimation,
        )

        return self._format_tensor_result(result)

    def get_initial_guess(self, source, target):
        src_down = self.preprocess(source)
        tgt_down = self.preprocess(target)

        src_fpfh = self.compute_fpfh(src_down)
        tgt_fpfh = self.compute_fpfh(tgt_down)

        ransac_result = self.global_registration_ransac(
            src_down, tgt_down, src_fpfh, tgt_fpfh
        )

        return ransac_result

    def register(self, source, target, method=None, ransac_retries=5, log=True):
        method = method or self.multi_scale_icp

        best_ransac = None

        for attempt in range(ransac_retries):
            ransac_result = self.get_initial_guess(source, target)
            if log:
                logging.info(
                    "Attempt %d/%d — fitness: %.4f  RMSE: %.6f",
                    attempt + 1,
                    ransac_retries,
                    ransac_result.fitness,
                    ransac_result.inlier_rmse,
                )

            if best_ransac is None or ransac_result.fitness > best_ransac.fitness:
                best_ransac = ransac_result
        icp_result = method(source, target, best_ransac.transformation)

        # Use the same evaluation path as benchmark for consistent fitness numbers
        evaluation = self.evaluate_alignment(source, target, icp_result.transformation)

        return icp_result, ransac_result, evaluation

    def evaluate_alignment(self, source, target, transform):
        return o3d.pipelines.registration.evaluate_registration(
            source,
            target,
            transformation=transform,
            max_correspondence_distance=self.max_correspondence_distance,
        )

    def visualise_result(
        self, source, target=None, transform=np.eye(4), downsample=0.008, write=False
    ):  # Downsample sets voxel size: =1 gives one voxel for the whole point cloud
        if target is not None:

            source.paint_uniform_color([1, 0.2, 0])
            target.paint_uniform_color([0, 0.65, 0.93])

            source.transform(transform)

            if write:
                o3d.io.write_point_cloud(
                    "../data/debug/reg_cloud.ply",
                    source + target,
                )

            src_d = self.downsample(source, ratio=downsample)
            tgt_d = self.downsample(target, ratio=downsample)
            title = f"Alignment after transformation with {transform}"

            o3d.visualization.draw_geometries(
                [src_d, tgt_d],
                window_name=title,
                width=1600,
                height=1000,
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
            # results.append(self.benchmark_method(self.icp, src, tgt, init_guess))
            # results.append(self.benchmark_method(self.plane_icp, src, tgt, init_guess))
            # results.append(self.benchmark_method(self.gen_icp, src, tgt, init_guess))
            results.append(
                self.benchmark_method(self.multi_scale_icp, src, tgt, init_guess)
            )

        # 3. Print the unified table
        self.print_result_summary(results)

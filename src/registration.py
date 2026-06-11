import open3d as o3d
import open3d.core as o3c
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import copy
import time
import logging

from scipy.signal import savgol_filter
from scipy.interpolate import CubicSpline, Akima1DInterpolator, PchipInterpolator
from scipy.integrate import cumulative_trapezoid


# Full registration pipeline: velocity-correct raw profiler scans, preprocess,
# globally align with FPFH + RANSAC, then refine locally with multi-scale ICP.
class Registration:
    def __init__(self, voxel_size=0.05, fps=1000, C_d=0.1):
        self.voxel = voxel_size
        self.fps = fps
        self.C_d = (
            C_d  # Nominal inter-profile spacing (mm per profile at constant speed)
        )

        # Tensor device config — swapped to GPU here if available
        self.device = o3c.Device("CPU:0")
        self.float_dtype = o3c.float32

        # Axis permutation applied to profiler frame before registration
        self.tf = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]])

        # ICP correspondence distances scale with voxel to stay resolution-consistent
        self.max_correspondence_distance = self.voxel * 0.4

        # Normal estimation parameters
        self.normal_radius = self.voxel * 2
        self.normal_max_nn = 30

        # FPFH feature extraction parameters
        self.fpfh_radius = self.voxel * 5
        self.fpfh_max_nn = 100

        # RANSAC global registration parameters
        self.ransac_distance_threshold = self.voxel * 1.5
        self.ransac_max_iteration = 100_000
        self.ransac_confidence = 0.999

        self.pcd = o3d.geometry.PointCloud()

        # Shared ICP convergence criteria used by all local refinement methods
        self.relative_fitness = 1e-6
        self.relative_rmse = 1e-6
        self.max_iteration = 150
        self.criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
            relative_fitness=self.relative_fitness,
            relative_rmse=self.relative_rmse,
            max_iteration=self.max_iteration,
        )

    def load_pcd(self, file_path):
        """Loads point cloud from path (.ply)"""
        pcd = o3d.io.read_point_cloud(file_path)
        return pcd

    def _format_tensor_result(self, tensor_result):
        """Formats tensor-based output of CPU-accelerated registration in terms of legacy Open3D format"""
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
        """Verifies whether point cloud contains normals, and computes them if necessary"""
        if not pcd.has_normals():
            pcd.estimate_normals(
                o3d.geometry.KDTreeSearchParamHybrid(
                    radius=self.normal_radius,
                    max_nn=self.normal_max_nn,
                )
            )
        return pcd

    def simple_convert(self, file_path):
        """.STL to .ply simple conversion"""
        if file_path is not None:
            mesh = o3d.io.read_triangle_mesh(file_path)
            mesh.compute_vertex_normals()
            self.pcd.points = mesh.vertices
            self.pcd.normals = mesh.vertex_normals
        return self.pcd

    def poisson_convert(self, file, n_points=50000):
        """.STL to .ply conversion using Poisson disk sampling"""
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
        """Simple geometrical cropping of point cloud to remove regions of little interest"""
        xyz = np.asarray(pcd.points)
        heights = xyz[:, height_axis]
        widths = xyz[:, width_axis]

        floor_y = np.percentile(heights, 1) if robust_floor else np.min(heights)
        abs_thresh_y = floor_y + max_y_threshold

        floor_x = np.percentile(widths, 1) if robust_floor else np.min(widths)
        roof_x = np.percentile(widths, 99) if robust_floor else np.max(widths)
        min_thresh_x = floor_x + x_thresh
        max_thresh_x = roof_x - x_thresh

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
        """Sets voxel size based on dimensions of point cloud"""
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_dimension = np.max(extent)
        self.voxel = max_dimension * ratio
        logging.info(f"Max dimension: {max_dimension}; Voxel size: {self.voxel}")

    def velocity_correction(
        self,
        csv_or_df,
        pcd,
        method="akima",
        denoise=False,
        downsample_step=5,
        visualise=True,
    ):
        """
        Constructs geometrically accurate point cloud from profiles obtained from
        2D laser profiler. Input is a .csv file containing speed data of object as
        it moves past the laser profiler.
        """
        if isinstance(csv_or_df, str):
            df = pd.read_csv(csv_or_df)
        else:
            df = csv_or_df

        v_time = df["Time_s"].values
        v_speed = df["Speed_mms"].values

        if denoise:
            v_speed = savgol_filter(v_speed, window_length=101, polyorder=3)

        # Derive acquisition time window from the nominal y-extent of the raw cloud
        pcd_raw = np.asarray(pcd.points)
        y_min_raw = pcd_raw[:, 1].min()
        y_max_raw = pcd_raw[:, 1].max()

        t_0 = y_min_raw / (self.C_d * self.fps)
        t_e = y_max_raw / (self.C_d * self.fps)
        t_tot = t_e - t_0

        total_lines = int(np.floor(t_tot * self.fps)) + 1
        line_indices = np.arange(total_lines)
        t_line = t_0 + line_indices / self.fps

        # Downsample speed data before fitting to reduce spline overfitting,
        # then evaluate the fitted spline at each profile timestamp
        method = method.lower()
        if method in ["cubic", "akima", "pchip"] and len(v_time) > (
            downsample_step * 4
        ):
            sample_time = v_time[::downsample_step]
            sample_speed = v_speed[::downsample_step]

            if sample_time[-1] != v_time[-1]:
                sample_time = np.append(sample_time, v_time[-1])
                sample_speed = np.append(sample_speed, v_speed[-1])

            t_line_clipped = np.clip(t_line, sample_time[0], sample_time[-1])

            if method == "cubic":
                spline_func = CubicSpline(sample_time, sample_speed, bc_type="natural")
            elif method == "akima":
                spline_func = Akima1DInterpolator(sample_time, sample_speed)
            elif method == "pchip":
                spline_func = PchipInterpolator(sample_time, sample_speed)

            v_line = spline_func(t_line_clipped)

            if visualise:
                fig, ax = plt.subplots(1, 1)
                ax.plot(t_line_clipped, v_line, color="red")
                ax.scatter(sample_time, sample_speed)
                plt.show()
                plt.close(fig)
        else:
            v_line = np.interp(t_line, v_time, v_speed)

        # Integrate speed to obtain the corrected y-position of each profile
        y_line = cumulative_trapezoid(v_line, dx=1.0 / self.fps, initial=0)

        # Map each raw point to its profile index and overwrite its y-coordinate
        y_min = t_0 * self.fps * self.C_d
        y_max = t_e * self.fps * self.C_d

        pcd_y_old = pcd_raw[:, 1]
        bus_mask = (pcd_y_old >= y_min) & (pcd_y_old <= y_max)

        if np.all(bus_mask):
            pcd_bus = pcd_raw.copy()
        else:
            pcd_bus = pcd_raw[bus_mask]

        PC_line_indices = np.round((pcd_bus[:, 1] - y_min) / self.C_d).astype(np.int32)
        np.clip(PC_line_indices, 0, total_lines - 1, out=PC_line_indices)

        pcd_bus[:, 1] = y_line[PC_line_indices]

        PC_corrected_o3d = o3d.geometry.PointCloud()
        PC_corrected_o3d.points = o3d.utility.Vector3dVector(pcd_bus)
        PC_corrected_o3d.paint_uniform_color([1, 0.2, 0])

        return PC_corrected_o3d

    def preprocess(self, pcd):
        """Point cloud preprocessing before FPFH + RANSAC global registration"""
        pcd_down = pcd.voxel_down_sample(self.voxel)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=self.normal_radius,
                max_nn=self.normal_max_nn,
            )
        )
        return pcd_down

    def SOR(self, pcd, neigbours, std_ratio):
        """Removes outliers from point cloud using Statistical Outlier Removal"""
        filtered_cloud, ind = pcd.remove_statistical_outlier(
            nb_neighbors=neigbours, std_ratio=std_ratio
        )
        removed = len(pcd.points) - len(filtered_cloud.points)
        return filtered_cloud, removed

    def radius_outlier_removal(self, pcd, n_points, radius):
        """Removes outliers from point cloud using Radius Outlier Removal"""
        filtered_cloud, ind = pcd.remove_radius_outlier(
            nb_points=n_points, radius=radius
        )
        removed = len(pcd.points) - len(filtered_cloud.points)
        return filtered_cloud, removed

    def downsample(self, pcd, ratio):
        """Downsample point cloud based on ratio of max dimension"""
        bbox = pcd.get_axis_aligned_bounding_box()
        extent = bbox.get_extent()
        max_dimension = np.max(extent)
        dynamic_voxel = max_dimension * ratio

        pcd_down = pcd.voxel_down_sample(voxel_size=dynamic_voxel)
        pcd_down.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=dynamic_voxel * 2.5,
                max_nn=self.normal_max_nn,
            )
        )
        return pcd_down

    def compute_fpfh(self, pcd):
        """Computes Fast Point Feature Histograms on downsampled point cloud"""
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
        """Globally registers point clouds by RANSAC based on FPFH feature matching"""
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

    # --- Local refinement methods — all accept an initial transform from RANSAC ---

    def icp(self, source, target, init_transform):
        """Point-to-point ICP local refinement"""
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
        """Point-to-plane ICP with Tukey robust loss for outlier tolerance"""
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
        """Generalised ICP local refinement"""
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
        """
        Coarse-to-fine ICP on the tensor API: one pass at full voxel size,
        then a second at quarter-voxel for fine-grained refinement.
        """
        source = self.ensure_normals(source)
        target = self.ensure_normals(target)

        t_device = getattr(self, "device", o3c.Device("CPU:0"))
        t_dtype = getattr(self, "float_dtype", o3c.float32)

        t_src = o3d.t.geometry.PointCloud.from_legacy(source, t_dtype, t_device)
        t_tgt = o3d.t.geometry.PointCloud.from_legacy(target, t_dtype, t_device)

        estimation = o3d.t.pipelines.registration.TransformationEstimationPointToPlane()

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
        """Obtains global registration result via FPFH + RANSAC"""
        src_down = self.preprocess(source)
        tgt_down = self.preprocess(target)
        src_fpfh = self.compute_fpfh(src_down)
        tgt_fpfh = self.compute_fpfh(tgt_down)
        ransac_result = self.global_registration_ransac(
            src_down, tgt_down, src_fpfh, tgt_fpfh
        )
        return ransac_result

    def register(self, source, target, method=None, ransac_retries=5, log=True):
        """
        Two-stage pipeline: repeated RANSAC to find the best global alignment,
        followed by local ICP refinement from that initial guess.
        """
        method = method or self.multi_scale_icp
        best_ransac = None

        for attempt in range(ransac_retries):
            ransac_result = self.get_initial_guess(source, target)
            if log:
                logging.info(
                    "Attempt %d/%d - fitness: %.4f  RMSE: %.6f",
                    attempt + 1,
                    ransac_retries,
                    ransac_result.fitness,
                    ransac_result.inlier_rmse,
                )
            if best_ransac is None or ransac_result.fitness > best_ransac.fitness:
                best_ransac = ransac_result

        icp_result = method(source, target, best_ransac.transformation)

        # Re-evaluate on the dense clouds for metrics consistent with benchmark
        evaluation = self.evaluate_alignment(source, target, icp_result.transformation)
        return icp_result, ransac_result, evaluation

    def evaluate_alignment(self, source, target, transform):
        """Evaluates alignment between source and target after applying transform"""
        return o3d.pipelines.registration.evaluate_registration(
            source,
            target,
            transformation=transform,
            max_correspondence_distance=self.max_correspondence_distance,
        )

    # --- Visualisation ---

    def visualise_result(
        self, source, target=None, transform=np.eye(4), downsample=0.008, write=False
    ):
        """Visualises source, or source + target after applying a registration transform"""
        if target is not None:
            source.paint_uniform_color([1, 0.2, 0])
            target.paint_uniform_color([0, 0.65, 0.93])
            source.transform(transform)

            if write:
                o3d.io.write_point_cloud("../data/debug/reg_cloud.ply", source + target)

            src_d = self.downsample(source, ratio=downsample)
            tgt_d = self.downsample(target, ratio=downsample)
            o3d.visualization.draw_geometries(
                [src_d, tgt_d],
                window_name=f"Alignment after transformation with {transform}",
                width=1600,
                height=1000,
            )
        else:
            src_d = self.downsample(source, ratio=downsample)
            src_d.paint_uniform_color([1, 0.2, 0])
            src_d.transform(transform)
            o3d.visualization.draw_geometries(
                [src_d],
                window_name=f"Point cloud visualisation {transform}",
                width=1000,
                height=800,
            )

    # --- Benchmarking ---

    def rank_results(self, results):
        """Ranks registration methods by fitness, RMSE, and runtime"""
        successful = [r for r in results if r.get("success", False)]
        failed = [r for r in results if not r.get("success", False)]
        successful.sort(key=lambda r: (-r["fitness"], r["inlier_rmse"], r["runtime_s"]))
        return successful + failed

    def benchmark_method(self, method_fn, source, target, init_guess):
        """Runs one local ICP variant from a fixed initial guess and records metrics"""
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
        """Runs FPFH + RANSAC global registration and records metrics"""
        start = time.perf_counter()
        try:
            result = self.get_initial_guess(source, target)
            runtime_s = time.perf_counter() - start
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
        """Prints a ranked benchmark table of fitness, RMSE, and runtime per method"""
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
            print(
                f"{idx:<6}{item['method']:<24}{fitness:<14}"
                f"{rmse:<16}{item['runtime_s']:.6f:<14}{item['threshold']:<12.4f}{item['max_iter']:<10}"
            )
            if item["notes"]:
                print(f"      Notes: {item['notes']}")

    def benchmark(self, src, tgt):
        """
        Runs the full benchmark suite: global RANSAC first to get an initial
        transform, then each enabled local ICP variant from that same starting point.
        """
        results = []

        global_benchmark = self.benchmark_global_method(src, tgt)
        results.append(global_benchmark)

        init_guess = global_benchmark["transformation"]

        if global_benchmark["success"]:
            # Uncomment variants to include them in the comparison table
            # results.append(self.benchmark_method(self.icp, src, tgt, init_guess))
            # results.append(self.benchmark_method(self.plane_icp, src, tgt, init_guess))
            # results.append(self.benchmark_method(self.gen_icp, src, tgt, init_guess))
            results.append(
                self.benchmark_method(self.multi_scale_icp, src, tgt, init_guess)
            )

        self.print_result_summary(results)

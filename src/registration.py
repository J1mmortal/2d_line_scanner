import copy
import time
import open3d as o3d
import numpy as np
from global_reg import Registration
import pygicp
from types import SimpleNamespace

reg = Registration()
threshold = reg.max_correspondence_distance
max_iter = reg.max_iteration
show_visuals = True


demo_pcs = o3d.data.DemoICPPointClouds()
#src = o3d.io.read_point_cloud(demo_pcs.paths[0])
#tgt = o3d.io.read_point_cloud(demo_pcs.paths[1])

tgt = reg.convert_file("data/Reg_block_2.STL", n_points=50000)
src = reg.convert_file("data/Reg_block_dented.STL", n_points=50000)

# #### Initial guess for transformation when global method not working ####
# init_guess = np.asarray([
#     [0.862, 0.011, -0.507, 0.5],
#     [-0.139, 0.967, -0.215, 0.7],
#     [0.487, 0.255, 0.835, -1.4],
#     [0.0, 0.0, 0.0, 1.0]
# ])


def preprocess_cloud(pc, voxel_size=None):
    pc = copy.deepcopy(pc)
    if voxel_size is not None:
        pc = pc.voxel_down_sample(voxel_size)
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=reg.normal_radius,
            max_nn=reg.normal_max_nn,
        )
    )
    return pc

def ensure_normals(pc):
    pc = copy.deepcopy(pc)
    if not pc.has_normals():
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=reg.normal_radius,
                max_nn=reg.normal_max_nn,
            )
        )
    return pc

def pcd_to_numpy(pc):
    return np.asarray(pc.points, dtype=np.float64)

def make_icp_criteria(max_iter):
    return o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=reg.relative_fitness, relative_rmse=reg.relative_rmse, max_iteration=max_iter)

def icp_refinement(source_pc, target_pc, init_guess, threshold=1.0, max_iter=100):
    source_pc = ensure_normals(source_pc)
    target_pc = ensure_normals(target_pc)
    criteria = make_icp_criteria(max_iter)
    return o3d.pipelines.registration.registration_icp(source_pc, target_pc, threshold, init_guess, o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)

def point_to_plane_refinement(source_pc, target_pc, init_guess, threshold=1.0, max_iter=100):
    source_pc = ensure_normals(source_pc)
    target_pc = ensure_normals(target_pc)
    criteria = make_icp_criteria(max_iter)
    return o3d.pipelines.registration.registration_icp(source_pc, target_pc, threshold, init_guess, o3d.pipelines.registration.TransformationEstimationPointToPlane(), criteria)

def gicp_refinement(source_pc, target_pc, init_guess, threshold=1.0, max_iter=100):
    source_pc = ensure_normals(source_pc)
    target_pc = ensure_normals(target_pc)
    criteria = make_icp_criteria(max_iter)
    return o3d.pipelines.registration.registration_generalized_icp(source_pc, target_pc, threshold, init_guess, o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(), criteria)

def vgicp_refinement(source_pc, target_pc, init_guess, threshold=1.0, max_iter=100):
    source_np = pcd_to_numpy(source_pc)
    target_np = pcd_to_numpy(target_pc)
    transform = pygicp.align_points(target_np, source_np, initial_guess=init_guess, method="VGICP", max_correspondence_distance=threshold, voxel_resolution=reg.coarse_voxel, num_threads=4)
    return SimpleNamespace(transformation=transform)

def evaluate_alignment(source_pc, target_pc, transform, threshold):
    return o3d.pipelines.registration.evaluate_registration(source_pc, target_pc, threshold, transform)

def visualise_result(source_pc, target_pc, transform, title="Results for unassigned method"):
    source_temp = copy.deepcopy(source_pc)
    target_temp = copy.deepcopy(target_pc)
    source_temp.paint_uniform_color([1, 0.2, 0])
    target_temp.paint_uniform_color([0, 0.65, 0.93])
    source_temp.transform(transform)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp],
        window_name=title,
        width=1000,
        height=800)

def benchmark_method(name, method_fn, source_pc, target_pc, init_guess, threshold, max_iter):
    start = time.perf_counter()
    result = method_fn(source_pc, target_pc, init_guess, threshold, max_iter)
    runtime_s = time.perf_counter() - start
    if result is None:
        return {
            "method": name,
            "success": False,
            "fitness": None,
            "inlier_rmse": None,
            "runtime_s": runtime_s,
            "threshold": threshold,
            "max_iter": max_iter,
            "relative_fitness": reg.relative_fitness,
            "relative_rmse": reg.relative_rmse,
            "transformation": None,
            "result": None,
            "evaluation": None,
            "notes": "Method not implemented."}

    evaluation = evaluate_alignment(source_pc, target_pc, result.transformation, threshold)
    return {
        "method": name,
        "success": True,
        "fitness": evaluation.fitness,
        "inlier_rmse": evaluation.inlier_rmse,
        "runtime_s": runtime_s,
        "threshold": threshold,
        "max_iter": max_iter,
        "relative_fitness": reg.relative_fitness,
        "relative_rmse": reg.relative_rmse,
        "transformation": result.transformation,
        "result": result,
        "evaluation": evaluation,
        "notes": ""}

def rank_results(results):
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    successful.sort(key=lambda r: (-r["fitness"], r["inlier_rmse"], r["runtime_s"]))
    return successful + failed

def print_initial_evaluation(source_pc, target_pc, init_guess, threshold, show_visuals=False):
    print("Before registration:")
    initial_eval = evaluate_alignment(source_pc, target_pc, init_guess, threshold)
    print(initial_eval)
    if show_visuals:
        visualise_result(source_pc, target_pc, init_guess, title="Initial guess")
    return initial_eval

def print_result_summary(results):
    print("===== Ranked benchmark summary =====")
    header = (f"{'Rank':<6}{'Method':<24}{'Fitness':<14}"
              f"{'Inlier RMSE':<16}{'Runtime (s)':<14}{'Threshold':<12}{'Max iter':<10}")
    print(header)
    print("-" * len(header))

    for idx, item in enumerate(results, start=1):
        fitness = f"{item['fitness']:.6f}" if item["fitness"] is not None else "N/A"
        rmse = f"{item['inlier_rmse']:.6f}" if item["inlier_rmse"] is not None else "N/A"
        runtime_s = f"{item['runtime_s']:.6f}"
        print(
            f"{idx:<6}{item['method']:<24}{fitness:<14}"
            f"{rmse:<16}{runtime_s:<14}{item['threshold']:<12.4f}{item['max_iter']:<10}")
        if item["notes"]:
            print(f"      Notes: {item['notes']}")

def visualise_ranked_results(results, source_pc, target_pc, show_visuals=False):
    if not show_visuals:
        return
    for item in results:
        if item["success"] and item["transformation"] is not None:
            visualise_result(source_pc, target_pc, item["transformation"], title=item["method"])

def main(show_visuals):
    preprocess_cloud(src, voxel_size=reg.coarse_voxel)
    preprocess_cloud(tgt, voxel_size=reg.coarse_voxel)
    global_result = reg.get_initial_guess(src, tgt)
    init_guess = global_result.transformation
    initial_eval = print_initial_evaluation(src, tgt, init_guess, threshold, show_visuals=show_visuals)
    
    methods = [
        ("ICP (point-to-point)", icp_refinement),
        ("ICP (point-to-plane)", point_to_plane_refinement),
        ("G-ICP", gicp_refinement),
        ("VG-ICP", vgicp_refinement)]
    
    benchmark_results = [benchmark_method(name, fn, src, tgt, init_guess, threshold, max_iter) for name, fn in methods]
    ranked_results = rank_results(benchmark_results)
    print("After registration:")
    for item in ranked_results:
        print(f"\n\n===== {item['method']} =====")
        if item["success"]:
            print(item["result"])
            print(item["transformation"])
            print(item["evaluation"])
        else:
            print(item["notes"])
    
    print_result_summary(ranked_results)
    visualise_ranked_results(ranked_results, src, tgt, show_visuals=show_visuals)
    return {
        "initial_evaluation": initial_eval,
        "initial_guess": init_guess,
        "results": benchmark_results,
        "ranked_results": ranked_results}

if __name__ == "__main__":
    summary = main(show_visuals=show_visuals)
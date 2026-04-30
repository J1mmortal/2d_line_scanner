import copy
import time
import open3d as o3d
import numpy as np
import sys
from pathlib import Path
from types import SimpleNamespace

SRC_DIR = Path(r"\\wsl.localhost\Ubuntu\home\jim\Codes\2d_line_scanner\src")
DARE_repo = Path(r"\\wsl.localhost\Ubuntu\home\jim\dare\DARE")
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DARE_repo))

from start import psreg
from start import observation_weights
from registration import Registration

reg = Registration(3, 3)
threshold = reg.max_correspondence_distance
max_iter = reg.max_iteration
show_visuals = False
density_aware = True

demo_pcs = o3d.data.DemoICPPointClouds()
# src = o3d.io.read_point_cloud(demo_pcs.paths[0])
# tgt = o3d.io.read_point_cloud(demo_pcs.paths[1])

# tgt = reg.convert_file_data("data/Reg_block_2.STL", n_points=50000)
# src = reg.convert_file_data("data/Reg_block_tripledented.STL", n_points=50000)

tgt = o3d.io.read_point_cloud("data/Reg_block_2_smooth.ply")
src = o3d.io.read_point_cloud("data/Reg_block_tripledented_abrupt_50000.ply")

#### Initial guess for transformation when global method not working ####
# init_guess = np.asarray([
#     [0.862, 0.011, -0.507, 0.5],
#     [-0.139, 0.967, -0.215, 0.7],
#     [0.487, 0.255, 0.835, -1.4],
#     [0.0, 0.0, 0.0, 1.0]
# ])

timings = []


def timed_step(name, fn, *args, **kwargs):
    """Run fn(*args, **kwargs), measure time, log and print it."""
    start = time.perf_counter()
    result = fn(*args, **kwargs)
    duration = time.perf_counter() - start
    timings.append({"name": name, "duration": duration})
    print(f"[TIMING] {name} took {duration:.3f} s")
    return result


def preprocess_cloud(pc, voxel_size=None):
    pc = copy.deepcopy(pc)
    if voxel_size is not None:
        pc = pc.voxel_down_sample(voxel_size)
    pc.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(
            radius=reg.normal_radius, max_nn=reg.normal_max_nn
        )
    )
    return pc


def ensure_normals(pc):
    pc = copy.deepcopy(pc)
    if not pc.has_normals():
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=reg.normal_radius, max_nn=reg.normal_max_nn
            )
        )
    return pc


def pcd_to_numpy(pc):
    return np.asarray(pc.points, dtype=np.float64).T


def make_icp_criteria(max_iter):
    return o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=reg.relative_fitness,
        relative_rmse=reg.relative_rmse,
        max_iteration=max_iter,
    )


def dare_refinement(
    source_pc,
    target_pc,
    init_guess,
    num_iters=30,
    K=None,
    K_max=300,
    gamma=0.005,
    epsilon=1e-5,
    num_neighbors=10,
    points_per_comp=200,
):
    Vs = [pcd_to_numpy(source_pc), pcd_to_numpy(target_pc)]
    features = []
    if K is None:
        N = Vs[0].shape[1] + Vs[1].shape[1]
        K = min(max(10, N // points_per_comp), K_max)
    pk = psreg.get_default_cluster_priors(K, gamma)
    X = psreg.get_randn_cluster_means(Vs, K)
    Q = psreg.get_default_cluster_precisions(Vs, X)
    beta = psreg.get_default_beta(Q, gamma)

    # source starts from initial guess, target fixed at identity
    R_src = init_guess[:3, :3].copy()
    t_src = init_guess[:3, 3].copy()
    R_tgt = np.eye(3)
    t_tgt = np.zeros(3)
    Ps = [(R_src, t_src), (R_tgt, t_tgt)]

    method = psreg.PSREG(beta, epsilon, pk, X, Q, [], debug=False)
    TVs, X_model = method.register_points(
        Vs,
        features,
        num_iters,
        Ps,
        show_progress=True,
        observation_weight_function=observation_weights.empirical_estimate_kdtree,
        ow_args=num_neighbors,
    )

    # print("type(TVs[0]) =", type(TVs[0]), "shape =", np.asarray(TVs[0]).shape)
    # print("Ps[0] type:", type(Ps[0]), "len:", len(Ps[0]))
    # print("Ps[0][0] shape:", np.asarray(Ps[0][0]).shape)
    # print("Ps[0][1] shape:", np.asarray(Ps[0][1]).shape)

    R_refined, t_refined = Ps[0]
    T_source_refined = np.eye(4)
    T_source_refined[:3, :3] = R_refined
    T_source_refined[:3, 3] = t_refined
    return SimpleNamespace(
        transformation=T_source_refined,
        result=TVs,
        model=X_model,
        fitness=None,
        inlier_rmse=None,
    )


def icp_refinement(source_pc, target_pc, init_guess, threshold=1.0, max_iter=100):
    source_pc = ensure_normals(source_pc)
    target_pc = ensure_normals(target_pc)
    criteria = make_icp_criteria(max_iter)
    return o3d.pipelines.registration.registration_icp(
        source_pc,
        target_pc,
        threshold,
        init_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria,
    )


def point_to_plane_refinement(
    source_pc, target_pc, init_guess, threshold=1.0, max_iter=100
):
    source_pc = ensure_normals(source_pc)
    target_pc = ensure_normals(target_pc)
    criteria = make_icp_criteria(max_iter)
    return o3d.pipelines.registration.registration_icp(
        source_pc,
        target_pc,
        threshold,
        init_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria,
    )


def gicp_refinement(source_pc, target_pc, init_guess, threshold=1.0, max_iter=100):
    source_pc = ensure_normals(source_pc)
    target_pc = ensure_normals(target_pc)
    criteria = make_icp_criteria(max_iter)
    return o3d.pipelines.registration.registration_generalized_icp(
        source_pc,
        target_pc,
        threshold,
        init_guess,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria,
    )


def evaluate_alignment(source_pc, target_pc, transform, threshold):
    return o3d.pipelines.registration.evaluate_registration(
        source_pc, target_pc, threshold, transform
    )


def visualise_result(
    source_pc, target_pc, transform, title="Results for unassigned method"
):
    source_temp = copy.deepcopy(source_pc)
    target_temp = copy.deepcopy(target_pc)
    source_temp.paint_uniform_color([1, 0.2, 0])
    target_temp.paint_uniform_color([0, 0.65, 0.93])
    source_temp.transform(transform)
    o3d.visualization.draw_geometries(
        [source_temp, target_temp], window_name=title, width=1000, height=800
    )


def benchmark_method(
    name, method_fn, source_pc, target_pc, init_guess, threshold, max_iter
):
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
            "notes": "Method not implemented.",
        }

    evaluation = evaluate_alignment(
        source_pc, target_pc, result.transformation, threshold
    )
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
        "notes": "",
    }


def rank_results(results):
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]
    successful.sort(key=lambda r: (-r["fitness"], r["inlier_rmse"], r["runtime_s"]))
    return successful + failed


def print_initial_evaluation(
    source_pc, target_pc, init_guess, threshold, show_visuals=False
):
    print("After global registration:")
    initial_eval = evaluate_alignment(source_pc, target_pc, init_guess, threshold)
    print(initial_eval)
    if show_visuals:
        visualise_result(source_pc, target_pc, init_guess, title="Initial guess")
    return initial_eval


def print_dare_evaluation(
    source_pc, target_pc, guess_refinement, threshold, show_visuals=False
):
    print("After DARE refinement:")
    dare_eval = evaluate_alignment(source_pc, target_pc, guess_refinement, threshold)
    print(dare_eval)
    if show_visuals:
        visualise_result(
            source_pc, target_pc, guess_refinement, title="DARE refinement"
        )
    return dare_eval


def print_result_summary(results):
    print("===== Ranked benchmark summary =====")
    header = (
        f"{'Rank':<6}{'Method':<24}{'Fitness':<14}"
        f"{'Inlier RMSE':<16}{'Runtime (s)':<14}{'Threshold':<12}{'Max iter':<10}"
    )
    print(header)
    print("-" * len(header))

    for idx, item in enumerate(results, start=1):
        fitness = f"{item['fitness']:.6f}" if item["fitness"] is not None else "N/A"
        rmse = (
            f"{item['inlier_rmse']:.6f}" if item["inlier_rmse"] is not None else "N/A"
        )
        runtime_s = f"{item['runtime_s']:.6f}"
        print(
            f"{idx:<6}{item['method']:<24}{fitness:<14}"
            f"{rmse:<16}{runtime_s:<14}{item['threshold']:<12.4f}{item['max_iter']:<10}"
        )
        if item["notes"]:
            print(f"      Notes: {item['notes']}")


def visualise_ranked_results(results, source_pc, target_pc, show_visuals=False):
    if not show_visuals:
        return
    for item in results:
        if item["success"] and item["transformation"] is not None:
            visualise_result(
                source_pc, target_pc, item["transformation"], title=item["method"]
            )


def main(show_visuals):
    global timings
    total_start = time.perf_counter()

    src_proc = timed_step(
        "preprocess src (coarse)", preprocess_cloud, src, voxel_size=reg.coarse_voxel
    )
    print("src points:", np.asarray(src.points).shape)
    print("src points (processed):", np.asarray(src_proc.points).shape)

    tgt_proc = timed_step(
        "preprocess tgt (coarse)", preprocess_cloud, tgt, voxel_size=reg.coarse_voxel
    )
    print("tgt points:", np.asarray(tgt.points).shape)
    print("tgt points (processed):", np.asarray(tgt_proc.points).shape)

    """
    print("src points:", np.asarray(src_proc.points).shape)
    print("tgt points:", np.asarray(tgt_proc.points).shape)
    
    global_result = reg.get_initial_guess(src_proc, tgt_proc)
    init_guess = global_result.transformation
    initial_eval = print_initial_evaluation(src_proc, tgt_proc, init_guess, threshold, show_visuals=False)
    
    print("--- Testing ICP point-to-point ---")
    r1 = icp_refinement(src_proc, tgt_proc, init_guess, threshold, max_iter)
    print("Done:", r1.fitness)
    
    print("--- Testing ICP point-to-plane ---")
    r2 = point_to_plane_refinement(src_proc, tgt_proc, init_guess, threshold, max_iter)
    print("Done:", r2.fitness)
    
    print("--- Testing G-ICP ---")
    r3 = gicp_refinement(src_proc, tgt_proc, init_guess, threshold, max_iter)
    print("Done:", r3.fitness)
    
    print("--- Testing VG-ICP ---")
    r4 = vgicp_refinement(src_proc, tgt_proc, init_guess, threshold, max_iter)
    print("Done:", r4.transformation)
    """
    if density_aware == True:
        global_result = timed_step(
            "global initial guess (RANSAC+FPFH)", reg.get_initial_guess, src, tgt
        )
        init_guess = global_result.transformation
        initial_eval = timed_step(
            "evaluate + visualise global result",
            print_initial_evaluation,
            src,
            tgt,
            init_guess,
            threshold,
            show_visuals=show_visuals,
        )
        dare_result = timed_step(
            "DARE refinement",
            dare_refinement,
            src_proc,
            tgt_proc,
            init_guess,
            num_iters=15,
            points_per_comp=500,
        )
        guess_refinement = dare_result.transformation
        dare_eval = timed_step(
            "evaluate + visualise DARE result",
            print_dare_evaluation,
            src,
            tgt,
            guess_refinement,
            threshold,
            show_visuals=show_visuals,
        )
        guess = guess_refinement
    else:
        global_result = timed_step(
            "global initial guess (RANSAC+FPFH)", reg.get_initial_guess, src, tgt
        )
        init_guess = global_result.transformation
        initial_eval = timed_step(
            "evaluate + visualise global result",
            print_initial_evaluation,
            src,
            tgt,
            init_guess,
            threshold,
            show_visuals=show_visuals,
        )
        guess = init_guess
        guess_refinement = None
        dare_eval = None

    methods = [
        ("ICP (point-to-point)", icp_refinement),
        ("ICP (point-to-plane)", point_to_plane_refinement),
        ("G-ICP", gicp_refinement),
    ]

    def _run_benchmarks():
        return [
            benchmark_method(name, fn, src, tgt, guess, threshold, max_iter)
            for name, fn in methods
        ]

    benchmark_results = timed_step("all local methods (benchmarks)", _run_benchmarks)
    ranked_results = rank_results(benchmark_results)

    def _print_local_results():
        ranked = rank_results(benchmark_results)
        print("After local registration:")
        for item in ranked:
            print(f"\n\n===== {item['method']} =====")
            if item["success"]:
                print(item["result"])
                print(item["transformation"])
                print(item["evaluation"])
            else:
                print(item["notes"])
        return ranked

    ranked_results = timed_step("print local method details", _print_local_results)

    def _print_timing_summary():
        print("\n===== Timing summary (top 10 by duration) =====")
        sorted_steps = sorted(timings, key=lambda d: d["duration"], reverse=True)
        top = sorted_steps[:10]
        for i, entry in enumerate(top, 1):
            print(f"{i:2d}. {entry['name']:<40} {entry['duration']:.3f} s")
        total = sum(entry["duration"] for entry in sorted_steps)
        print(f"\nTotal measured time (sum of steps): {total:.3f} s")

    timed_step("print timing summary", _print_timing_summary)
    timed_step("print benchmark summary", print_result_summary, ranked_results)

    total_runtime = time.perf_counter() - total_start
    print(f"\n===== TOTAL wall clock time =====\n{total_runtime:.3f} s")
    visualise_ranked_results(ranked_results, src, tgt, show_visuals=show_visuals)
    return {
        "initial_evaluation": initial_eval,
        "initial_guess": init_guess,
        "dare_evaluation": dare_eval,
        "dare_guess": guess_refinement,
        "results": benchmark_results,
        "ranked_results": ranked_results,
        "timings": timings,
        "total_runtime": total_runtime,
    }


if __name__ == "__main__":
    summary = main(show_visuals=show_visuals)

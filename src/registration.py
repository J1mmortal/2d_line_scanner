import open3d as o3d
import copy
from src.global_reg import Registration

reg = Registration()
threshold = reg.max_correspondence_distance
max_iter = reg.max_iteration
demo_pcs = o3d.data.DemoICPPointClouds()
src = o3d.io.read_point_cloud(demo_pcs.paths[0])
tgt = o3d.io.read_point_cloud(demo_pcs.paths[1])

# src = o3d.io.read_point_cloud("reference_scan.ply")
# tgt = o3d.io.read_point_cloud("dented_scan.ply")

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
            max_nn=reg.normal_max_nn
        )
    )
    return pc

def ensure_normals(pc):
    pc = copy.deepcopy(pc)
    if not pc.has_normals():
        pc.estimate_normals(
            o3d.geometry.KDTreeSearchParamHybrid(
                radius=reg.normal_radius,
                max_nn=reg.normal_max_nn
            )
        )
    return pc

def icp_refinement(source_pc, target_pc, init_guess, threshold=1.0, max_iter=100):
    source_pc = ensure_normals(source_pc)
    target_pc = ensure_normals(target_pc)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=reg.relative_fitness,
        relative_rmse=reg.relative_rmse,
        max_iteration=max_iter
    )

    result = o3d.pipelines.registration.registration_icp(
        source_pc,
        target_pc,
        threshold,
        init_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        criteria
    )
    return result

def point_to_plane_refinement(source_pc, target_pc, init_guess, threshold=1.0, max_iter=100):
    source_pc = ensure_normals(source_pc)
    target_pc = ensure_normals(target_pc)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=reg.relative_fitness,
        relative_rmse=reg.relative_rmse,
        max_iteration=max_iter
    )

    result = o3d.pipelines.registration.registration_icp(
        source_pc,
        target_pc,
        threshold,
        init_guess,
        o3d.pipelines.registration.TransformationEstimationPointToPlane(),
        criteria
    )
    return result

def Gicp_refinement(source_pc, target_pc, init_guess, threshold=1.0, max_iter=100):
    source_pc = ensure_normals(source_pc)
    target_pc = ensure_normals(target_pc)

    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
        relative_fitness=reg.relative_fitness,
        relative_rmse=reg.relative_rmse,
        max_iteration=max_iter
    )

    result = o3d.pipelines.registration.registration_generalized_icp(
        source_pc,
        target_pc,
        threshold,
        init_guess,
        o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
        criteria
    )
    return result

def VGicp_refinement(source_pc, target_pc, init_guess, threshold):
    return

def evaluate_alignment(source_pc, target_pc, transform, threshold):
    return o3d.pipelines.registration.evaluate_registration(
        source_pc, target_pc, threshold, transform
    )

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
        height=800
    )
    return


#### Work flow ####
# Keep preprocessed versions available for future experiments,
# but refinement below is done on the full clouds for consistency.
source = preprocess_cloud(src, voxel_size=reg.coarse_voxel)
target = preprocess_cloud(tgt, voxel_size=reg.coarse_voxel)

global_result = reg.get_initial_guess(src, tgt)
init_guess = global_result.transformation

print("Before registration:")
visualise_result(src, tgt, init_guess, title="Initial guess")
before_eval = evaluate_alignment(src, tgt, init_guess, threshold)
print(before_eval)

print("After registration:")
print("===== ICP (point-to-point) =====")
icp_result = icp_refinement(src, tgt, init_guess, threshold, max_iter)
print(icp_result)
print(icp_result.transformation)
print(evaluate_alignment(src, tgt, icp_result.transformation, threshold))
visualise_result(src, tgt, icp_result.transformation, title="ICP refinement")

print("===== ICP (point-to-plane) =====")
plane_result = point_to_plane_refinement(src, tgt, init_guess, threshold, max_iter)
print(plane_result)
print(plane_result.transformation)
print(evaluate_alignment(src, tgt, plane_result.transformation, threshold))
visualise_result(src, tgt, plane_result.transformation, title="Point-to-plane refinement")

print("===== G-ICP =====")
gicp_result = Gicp_refinement(src, tgt, init_guess, threshold, max_iter)
print(gicp_result)
print(gicp_result.transformation)
print(evaluate_alignment(src, tgt, gicp_result.transformation, threshold))
visualise_result(src, tgt, gicp_result.transformation, title="G-ICP refinement")
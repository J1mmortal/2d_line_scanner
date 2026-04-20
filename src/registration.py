import numpy as np
import open3d as o3d
import copy

demo_pcs = o3d.data.DemoICPPointClouds()
source = o3d.io.read_point_cloud(demo_pcs.paths[0]) # pc with 191397 points.
target = o3d.io.read_point_cloud(demo_pcs.paths[1]) # pc with 137833 points.
o3d.visualization.draw_plotly([source],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
 
o3d.visualization.draw_plotly([target],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])

#### Initial Guess for Transformation ####
init_guess = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

def preprocess_cloud(pc, voxel_size=None):
    pc = copy.deepcopy(pc)
    if voxel_size is not None:
        pc = pc.voxel_down_sample(voxel_size)
    pc.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=2.0 * (voxel_size or 1.0), max_nn=30))
    return

def icp_refinement(source_pc, target_pc, init_guess, threshold=1.0, max_iter=100):
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iter)
    result = o3d.pipelines.registration.registration_icp(source_pc, target_pc, threshold, init_guess, o3d.pipelines.registration.TransformationEstimationPointToPoint(), criteria)
    return result

def Gicp_refinement(source_pc, target_pc, init_guess, threshold):
    return

def VGicp_refinement(source_pc, target_pc, init_guess, threshold):
    return

def evaluate_alignment(source_pc, target_pc, transform, threshold):
    return o3d.pipelines.registration.evaluate_registration(source_pc, target_pc, threshold, transform)

def visualise_result(source_pc, target_pc, transform, title="Results for unassigned method"):
    source_temp = copy.deepcopy(source_pc)
    target_temp = copy.deepcopy(target_pc)
    source_temp.paint_uniform_color([1, 0.2, 0])
    target_temp.paint_uniform_color([0, 0.65, 0.93])
    source_temp.transform(transform)
    o3d.visualization.draw_plotly([source_temp, target_temp], window_name=title)
    return

src = o3d.io.read_point_cloud("reference_scan.ply")
tgt = o3d.io.read_point_cloud("dented_scan.ply")
o3d.visualization.draw_plotly([source],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])
 
o3d.visualization.draw_plotly([target],
                                  zoom=0.455,
                                  front=[-0.4999, -0.1659, -0.8499],
                                  lookat=[2.1813, 2.0619, 2.0999],
                                  up=[0.1204, -0.9852, 0.1215])

source = preprocess_cloud(source, voxel_size=0.5)
target = preprocess_cloud(target, voxel_size=0.5)
init_guess = np.asarray([[0.862, 0.011, -0.507, 0.5],
                         [-0.139, 0.967, -0.215, 0.7],
                         [0.487, 0.255, 0.835, -1.4], [0.0, 0.0, 0.0, 1.0]])

print("Before ICP:")
before_eval = evaluate_alignment(source, target, init_guess, threshold=1.0)
print(before_eval)

icp_result = icp_refinement(source, target, init_guess, threshold=1.0, max_iterations=100)

print("After ICP:")
print(icp_result)
print("Transformation:")
print(icp_result.transformation)

after_eval = evaluate_alignment(source, target, icp_result.transformation, threshold=1.0)
print(after_eval)

visualise_result(source, target, icp_result.transformation, title="ICP refinement")
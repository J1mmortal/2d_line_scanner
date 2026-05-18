import numpy as np
import open3d as o3d
import os
import sys

#np.set_printoptions(threshold=sys.maxsize)

def move_PC_y_coordinates(V_list, T_0, T_e, laser_fps, set_dist, point_cloud):
    
    #convert to np.array for quicker operations
    V = np.array(V_list)

    ##Step 1: speed transformation to speed per line
        #Calculate total number of lines based on the exact timeframe
    total_lines = int(np.floor((T_e - T_0) * laser_fps)) + 1

        #Create the array of line indices [0, 1, 2... Ln]
    all_line_indices = np.arange(total_lines)

        #Convert raw times of the velocity measurement to the line number where line 0 starts at T_0
    raw_line_indices = (V[:, 0] - T_0) * laser_fps
    speeds = V[:, 1]

        #Interpolate speeds natively using the FULL speed profile
    speeds_per_line = np.interp(all_line_indices, raw_line_indices, speeds)

    ##Step 2: changes speed to the distance between lines and the position of the lines
    distances_per_line = speeds_per_line / laser_fps

        # calculates the Y position per line (last one removed because the speed of the final line is unnecessary)
    Y_absolute = np.zeros(total_lines)
    Y_absolute[1:] = np.cumsum(distances_per_line[:-1])

    ##Step 3: grab the lines of the point-cloud by matching y-values
    
        # 1. Physically crop the point cloud to remove points outside our time window
        # Assuming raw Y-coordinates represent the line number (Time * FPS)
    min_y_allowed = T_0 * laser_fps * set_dist
    max_y_allowed = T_e * laser_fps * set_dist
    
    print(min_y_allowed,max_y_allowed)
        # Create a mask that is only True for points inside our time window
    valid_points_mask = (np.round(point_cloud[:, 1],4) >= min_y_allowed) & (np.round(point_cloud[:, 1],4) <= max_y_allowed)
    
        # Apply the mask to delete all points outside the window
    cropped_cloud = point_cloud[valid_points_mask]


        # 2. We now look at all rows, Column 1 (Y) natively in Nx3 format
    current_Y = np.round(cropped_cloud[:, 1],4)
    _, line_indices = np.unique(current_Y, return_inverse=True)

    if abs(line_indices[-1]-total_lines)>10:
        print("FPS, T_0, T_e, line_distance input wrongfully", ", filtered pointcloud contains {0} lines while expecting {1} lines".format(line_indices[-1], total_lines))

    line_indices = np.clip(line_indices, 0, total_lines - 1)

        #overwrite cropped_cloud Y values with calculated Y distance at line number
    cropped_cloud[:, 1] = Y_absolute[line_indices]
    
    return cropped_cloud, Y_absolute

if __name__ == "__main__":
    T_0, T_e, FPS, linedistance = 2.5, 5, 1000.0, 0.05
    V_mock = [(0.0, 100.0), (10,100)]
    total_mock_lines = int((T_e - T_0) * FPS) + 1

    ply_filepath = r"C:\Users\roman\Documents\uni\BEP\pointclouds\bus_v2.ply"
    pcd = o3d.io.read_point_cloud(ply_filepath)
    raw_scanned_cloud = np.asarray(pcd.points)

    
    corrected_cloud, Y_coords = move_PC_y_coordinates(V_mock, T_0, T_e, FPS, linedistance, raw_scanned_cloud)
        
    print("\n--- Correction Complete ---")
    print(f"Original point cloud shape: {raw_scanned_cloud.shape}")
    print(f"Cropped & Corrected shape: {corrected_cloud.shape}")

    print("\nOpening 3D Viewer...")
    pcd_viz = o3d.geometry.PointCloud()
    pcd_viz.points = o3d.utility.Vector3dVector(corrected_cloud) 
    pcd_viz.paint_uniform_color([0.1, 0.5, 0.8]) 
    o3d.visualization.draw_geometries(
        [pcd_viz], 
        window_name="Corrected Laser Scan",
        width=1024, 
        height=768
    )
import numpy as np
import open3d as o3d

from registration import Registration
from damage_detection import DamageDetector
from data_analysis import DataAnalysis

reg = Registration(5)
det = DamageDetector()
dt = DataAnalysis()


# Example Integration:
# Assuming 'scores' was returned from the previous run_velocity_monte_carlo function
# plot_monte_carlo_results(scores, noise_std=1.5)

# Setup execution context
if __name__ == "__main__":

    # v_csv = r"..\data\speed_test_values.csv"
    v_csv = r"..\data\bus_kinematics.csv"

    tgt_p = r"..\data\bus\bus_v2.ply"
    tgt = reg.load_pcd(tgt_p)
    tgt.paint_uniform_color([0.0, 1.0, 0.0])
    # tgt = det.select_bus_hull(tgt, eps=2.0, visualise=False)

    src_p = r"..\data\bus\snelheid_test2.ply"
    src = reg.load_pcd(src_p)

    # Run simulation with 50 iterations and a standard deviation of 1.5 mm/s on speed
    # rmse, fitness = dt.run_velocity_monte_carlo(
    #     src,
    #     tgt=tgt,
    #     csv_path=v_csv,
    #     params=sim_params,
    #     num_iterations=50,
    #     noise_std=0.5,
    # )

    # plot_monte_carlo_results(rmse, noise_std=0.5)
    # plot_monte_carlo_results(fitness, noise_std=0.5)

    # list of biases to test (e.g., odometer under-registering or over-registering)
    means = np.linspace(-0.3, 0.3, 5)

    # list of random noise levels to test
    stds = np.linspace(0.0, 0.3, 5)

    rmsegrid, fitnessgrid, ransacgrid = dt.run_noise_grid_sweep(
        src,
        tgt,
        v_csv,
        means,
        stds,
        mc_iterations=1,
        uniform_donwsample=True,
    )

    topleft = ransacgrid[0, 0]
    topright = ransacgrid[0, -1]
    bottomleft = ransacgrid[-1, 0]
    bottomright = ransacgrid[-1, -1]

    o3d.visualization.draw_geometries(
        [
            reg.downsample(topleft, ratio=0.002),
            reg.downsample(tgt, ratio=0.002),
        ]
    )
    o3d.visualization.draw_geometries(
        [
            reg.downsample(topright, ratio=0.002),
            reg.downsample(tgt, ratio=0.002),
        ]
    )
    o3d.visualization.draw_geometries(
        [
            reg.downsample(bottomleft, ratio=0.002),
            reg.downsample(tgt, ratio=0.002),
        ]
    )
    o3d.visualization.draw_geometries(
        [
            reg.downsample(bottomright, ratio=0.002),
            reg.downsample(tgt, ratio=0.002),
        ]
    )
    # o3d.visualization.draw_geometries([topleft, tgt])
    # o3d.visualization.draw_geometries([topright, tgt])
    # o3d.visualization.draw_geometries(bottomleft, tgt)
    # o3d.visualization.draw_geometries([bottomright, tgt])

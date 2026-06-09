import numpy as np
import open3d as o3d

import logging

logging.basicConfig(
    filename="../data/monte_carlo.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    force=True,
)

from registration import Registration
from damage_detection import DamageDetector
from data_analysis import DataAnalysis

reg = Registration(5)
det = DamageDetector()


# Example Integration:
# Assuming 'scores' was returned from the previous run_velocity_monte_carlo function
# plot_monte_carlo_results(scores, noise_std=1.5)

# Setup execution context
if __name__ == "__main__":

    tgt_test = "../data/bus/gt/bus4.ply"
    # tgt_test = "../data/bus/scaled_speed_bus4.ply"
    # src_test = "../data/bus/lighting/bus4_lightson.ply"

    src_test = "../data/bus/snelheid_test_bus4.ply"

    v_csv = "../data/bus/speed_files/speed_bus4.csv"

    pipe_params = {
        "velocity_scale": True,
        "downsample_reg": False,
        "sigma_thresh": 3.0,
        "percentile": 80.0,
        "damage_outliers": 200,
        "cluster_eps": 1.1,
        "cluster_min_samples": 250,
        "crop": True,
        "visualise": False,
        "write": False,
    }

    dt = DataAnalysis(pipe_params)

    # Run simulation with 50 iterations and a standard deviation of 1.5 mm/s on speed
    # rmse, fitness, fn, fp = dt.run_velocity_monte_carlo(
    #     src_path=src_test,
    #     tgt_path=tgt_test,
    #     csv_path=v_csv,
    #     num_iterations=50,
    #     noise_std=0.025,
    # )

    # dt.plot_monte_carlo_results(rmse, noise_std=0.025)
    # dt.plot_monte_carlo_results(fitness, noise_std=0.025)
    # dt.plot_monte_carlo_results(fn, noise_std=0.025)
    # dt.plot_monte_carlo_results(fp, noise_std=0.025)

    # list of biases to test (e.g., odometer under-registering or over-registering)
    # means = np.linspace(-0.5, 0.5, 3)
    means = np.linspace(-0.025, 0.025, 11)

    # list of random noise levels to test
    # stds = np.linspace(0.0, 0.5, 3)
    stds = np.linspace(0.0, 0.03, 6)

    # archive1 = dt.run_noise_grid_sweep(
    #     src_test, tgt_test, v_csv, means, stds, mc_iterations=3
    # )

    # Tweak and run this as many times as you want to change labels, fonts, or cmaps
    # archive = "../data/sweep_results/sweep_metrics_denoise.npz"
    # dt.plot_saved_sweep_results(archive)

    # # topleft = src_grid[0, 0]
    # # topright = src_grid[0, -1]
    # # bottomleft = src_grid[-1, 0]
    # # bottomright = src_grid[-1, -1]

    # # topleft_t = tgt_grid[0, 0]
    # # topright_t = tgt_grid[0, -1]
    # # bottomleft_t = tgt_grid[-1, 0]
    # # bottomright_t = tgt_grid[-1, -1]

    # c1 = clustr_grid[0, 0]
    # c2 = clustr_grid[0, -1]
    # c3 = clustr_grid[-1, 0]
    # c4 = clustr_grid[-1, -1]

    # # # o3d.visualization.draw_geometries([topleft, topleft_t])
    # # # o3d.visualization.draw_geometries([topright, topright_t])
    # # # o3d.visualization.draw_geometries([bottomleft, bottomleft_t])
    # # # o3d.visualization.draw_geometries([bottomright, bottomright_t])

    # o3d.visualization.draw_geometries([c1])
    # o3d.visualization.draw_geometries([c2])
    # o3d.visualization.draw_geometries([c3])
    # o3d.visualization.draw_geometries([c4])

    delays = np.linspace(-0.1, 0.1, 21)

    # archive = dt.run_noise_delay_sweep(
    #     src_test,
    #     tgt_test,
    #     v_csv,
    #     delays,
    #     mc_iterations=3,
    # )

    # archive = "../data/sweep_results/delay_sweep_metrics.npz"

    # dt.plot_saved_delay_results(archive)

    # delay_std = 0.03

    # rmse, fitness, fn, fp = dt.run_time_sync_monte_carlo(
    #     src_path=src_test,
    #     tgt_path=tgt_test,
    #     csv_path=v_csv,
    #     num_iterations=200,
    #     delay_std=delay_std,
    # )

    # dt.plot_monte_carlo_results(
    #     rmse_res=rmse, fitness_res=fitness, fn_res=fn, fp_res=fp, noise_std=delay_std
    # )

    epss = np.linspace(0.6, 2.0, 15)
    min_samples = np.arange(150, 400, 15, dtype=int)
    print(epss, min_samples)

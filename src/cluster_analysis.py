import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import logging

from scipy.spatial import cKDTree

log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

from data_analysis import DataAnalysis

dt = DataAnalysis()

gt = "../data/bus4_gt.parquet"
guess = "../data/damage_metrics.parquet"

dt.compare_cluster_runs(gt, guess, 5, compact_view=True)

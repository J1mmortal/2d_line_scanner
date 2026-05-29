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

correct = "../data/damage_metrics_2.parquet"
guess = "../data/damage_metrics.parquet"

dt.compare_cluster_runs(correct, guess, 5)

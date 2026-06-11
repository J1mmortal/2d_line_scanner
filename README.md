# Vehicle Damage Detection Pipeline

This repository contains a 3D point cloud processing pipeline designed to identify, cluster, and quantify surface deformations (dents) on buses from raw 2D laser profiler data.

---

## System Architecture

The pipeline processes raw data through five main stages:

1. **Velocity Correction:** Combines 2D profiler frames into a spatially accurate 3D point cloud using spline-interpolated speed profiles.
2. **Preprocessing & Segmentation:** Extracts the vehicle hull using DBSCAN and applies geometric cropping filters.
3. **Global & Local Registration:** Aligns source and target clouds using an initial FPFH + RANSAC global pass followed by multi-scale local ICP refinement.
4. **Damage Detection:** Isolates surface anomalies via bidirectional Cloud-to-Cloud (C2C) distance thresholding against a robust Median Absolute Deviation (MAD) noise floor.
5. **Clustering & Quantification:** Groups anomaly points into discrete regions using DBSCAN and extracts 2.5D metrics (projected area, max depth, perimeter, approximate volume).

---

## Repository Structure

* **`main.py`:** Primary execution file.
* **`pipeline.py`:** Contains the core `Pipeline` driver class that orchestrates execution.
* **`registration.py`:** Implements global RANSAC alignment, multi-scale tensor-based ICP, and kinematic velocity correction.
* **`damage_detection.py`:** Handles MAD-based noise floor estimation, bidirectional distance thresholding, and PCA-based convex hull metrics evaluation.
* **`data_analysis.py`:** Tests pipeline robustness by adding synthetic velocity noise and time delays.

---

## Environment Setup

Install the required dependencies via Conda using the provided configuration file:

```bash
conda env create -f env.yml
conda activate <env_name>

```

---

## Usage

Configure the pipeline options directly within the main execution script and run the pipeline:

```bash
python main.py

```

> **Note:** Most of the code is explained with comments placed throughout the code. Parameters are explained in the paper.

### Robustness Evaluation

To evaluate registration performance under noisy tracking data, execute the analytical parameters sweep in `data_analysis.py`:

```bash
python data_analysis.py

```

This conducts a 2D grid sweep across velocity noise and jitter variations, exporting results directly as a `.npz` archive and generating summary plots.
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

## Data Directory Structure

The pipeline expects a specific relative directory structure located one level above the repository root directory. Create the following structure before executing the pipeline:

```text
├── data/
│   ├── bus/
│   │   ├── gt/
│   │   │   └── bus4.ply
│   │   └── speed_files/
│   │       └── speed_bus4.csv
│   ├── figs/
│   │   └── hulls/
│   ├── sweep_results/
│   ├── bus4_gt.parquet
│   └── damage_metrics.parquet
└── src/  <-- (Repository Root)
    ├── main.py
    ├── pipeline.py
    └── ...

```

---

## Environment Setup & System Prerequisites

### Conda Environment

Install the Python dependencies via Conda using the provided configuration file:

```bash
conda env create -f env.yml
conda activate <env_name>

```

### System-Level Dependencies (LaTeX)

The plotting functionality in `data_analysis.py` uses the `scienceplots` library, which requires a system-level LaTeX distribution to render text.

* **Linux:** `sudo apt-get install texlive-latex-extra texlive-fonts-recommended cm-super`
* **macOS:** `brew install --cask mactex-no-gui`
* **Windows:** Install MiKTeX and ensure the binaries are added to the system PATH.

---

## Hardware Acceleration

By default, the multi-scale ICP step runs on the CPU (`CPU:0`). To utilize CUDA-compatible hardware for processing dense clouds, modify the device configuration inside `registration.py`:

```python
# Change from:
self.device = o3c.Device("CPU:0")

# To:
self.device = o3c.Device("CUDA:0")

```

---

## Usage

### Standard Execution

Configure pipeline options directly within `main.py` and execute the script:

```bash
python main.py

```

```python
from pipeline import Pipeline

pip = Pipeline(
    source_path="../data/bus/snelheid_test_bus4.ply",
    target_path="../data/bus/gt/bus4.ply",
    velocity_scale=True,
    voxel_size=5.0,
    sigma_thresh=3.0,
    cluster_eps=1.1,
    cluster_min_samples=250,
    visualise=True
)

metrics = pip.run()

```

### Robustness Evaluation

To evaluate registration performance under noisy tracking telemetry, run the analytical parameter sweep:

```bash
python data_analysis.py

```

This executes a 2D grid sweep across velocity noise bias and jitter variations, exports the results as a compressed `.npz` archive, and outputs publication-ready summary charts.

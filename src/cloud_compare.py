import subprocess
from pathlib import Path
import open3d as o3d
import laspy
import numpy as np
import matplotlib.pyplot as plt


class CloudCompare:
    def __init__(
        self,
        comp_path,
        ref_path,
        params_path="..\data\m3c2_params.txt",
        cc_path="C:\Program Files\CloudCompare\CloudCompare.exe",
    ):
        self.cc_path = cc_path
        self.comp_path = comp_path
        self.ref_path = ref_path
        self.params_path = params_path

        for path in [Path(self.comp_path), Path(self.ref_path), Path(self.cc_path)]:
            if not path.exists():
                raise FileNotFoundError(f"Missing file: {path}")

    def run_c2c(self):
        """
        Executes CloudCompare C2C distance calculation headlessly.
        Returns the path to the newly generated file.
        """

        # Build the CLI command. Order is strictly enforced.
        cmd = [
            self.cc_path,
            "-SILENT",  # Suppress the GUI
            "-C_EXPORT_FMT",
            "LAS",  # Output format. LAS/BIN preserve scalar fields.
            "-O",
            self.comp_path,  # File 1: The scan being inspected
            "-O",
            self.ref_path,  # File 2: The pristine reference
            "-c2c_dist",  # Execute nearest-neighbor distance
        ]

        try:
            # check=True forces Python to raise an exception if CC crashes
            subprocess.run(cmd, capture_output=True, text=True, check=True)

        except subprocess.CalledProcessError as e:
            print(f"CloudCompare Exit Code: {e.returncode}")
            print(f"STDOUT:\n{e.stdout}")
            print(f"STDERR:\n{e.stderr}")
            raise RuntimeError("CloudCompare C2C pipeline failed. See logs above.")

        # CC automatically saves the output in the directory of the first loaded file
        # The suffix varies slightly by CC version, but typically contains "C2C_DIST"
        print(
            f"Processing complete. Look in {Path(self.comp_path).parent} for the generated file."
        )

    def run_m3c2(self):
        """
        Executes CloudCompare M3C2 distance calculation headlessly.
        """

        cmd = [
            self.cc_path,
            "-SILENT",
            "-C_EXPORT_FMT",
            "LAS",
            "-O",
            self.ref_path,
            "-O",
            self.comp_path,
            "-M3C2",
            self.params_path,
        ]

        try:
            subprocess.run(cmd, capture_output=True, text=True, check=True)

        except subprocess.CalledProcessError as e:
            print(f"CloudCompare Exit Code: {e.returncode}")
            print(f"STDOUT:\n{e.stdout}")
            print(f"STDERR:\n{e.stderr}")
            raise RuntimeError("CloudCompare M3C2 pipeline failed. Review logs.")

        # CC saves the output in the directory of the compared file
        print(
            f"Processing complete. Check {Path(self.comp_path).parent} for the generated file."
        )

    def read_las_data(self, las_path: Path):
        if not las_path.exists():
            raise FileNotFoundError(f"File not found: {las_path}")

        # 1. Load data via laspy
        las = laspy.read(str(las_path))
        points = np.vstack((las.x, las.y, las.z)).transpose()

        # 2. Extract distance scalar field
        # The name varies. If this fails, read the terminal output to find the correct string.
        if "M3C2" in str(las_path):
            dimension_name = "M3C2 distance"
        elif "C2C" in str(las_path):
            dimension_name = "C2C absolute distances"
        else:
            raise RuntimeError("Incorrect file loaded")

        try:
            # laspy accesses dimensions as attributes or dictionary keys
            distances = getattr(las, dimension_name)
        except AttributeError:
            dims = list(las.point_format.dimension_names)
            print(f"Available dimensions in LAS file: {dims}")
            raise ValueError(
                f"Could not find dimension matching '{dimension_name}'. Update the string."
            )

        abs_dist = np.abs(distances)

        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)

        return pcd, abs_dist  # Is return of pcd needed?


# ---------------------------------------------------------
# Execution Example
# ---------------------------------------------------------
if __name__ == "__main__":

    reference_scan = r"C:\Users\fvsch\OneDrive\Documents\Code\TUDelft\Y3\BEP\2d_line_scanner\data\CC\sin_tgt.ply"
    compared_scan = r"C:\Users\fvsch\OneDrive\Documents\Code\TUDelft\Y3\BEP\2d_line_scanner\data\CC\sin_src_reg.ply"

    ccl = CloudCompare(compared_scan, reference_scan)
    ccl.run_c2c(reference_scan, compared_scan)
    ccl.run_m3c2(reference_scan, compared_scan)

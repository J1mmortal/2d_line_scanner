import subprocess
from pathlib import Path
import open3d as o3d
import laspy
import numpy as np
import matplotlib.pyplot as plt
import shutil


class CloudCompare:
    def __init__(
        self,
        comp_path: str,
        ref_path: str,
        params_path=r"..\data\m3c2_params_test_block.txt",
        cc_path=r"C:\Program Files\CloudCompare\CloudCompare.exe",
        output_dir=r"../data/las",
    ):
        self.cc_path = cc_path
        self.comp_path = comp_path
        self.ref_path = ref_path
        self.params_path = params_path
        self.output_dir = Path(output_dir)

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

        # Search for the generated file using a wildcard pattern
        comp_path_obj = Path(self.comp_path)
        search_dir = comp_path_obj.parent
        pattern = f"{comp_path_obj.stem}_C2C_DIST_*.las"

        matches = list(search_dir.glob(pattern))

        if not matches:
            raise FileNotFoundError(
                f"Expected output matching '{pattern}' not found in {search_dir}"
            )

        # If there are multiple runs, grab the newest file based on modification time
        cc_output_file = max(matches, key=lambda p: p.stat().st_mtime)

        if self.output_dir:
            self.output_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure the directory exists

            final_target = self.output_dir / cc_output_file.name

            if cc_output_file.exists():
                shutil.move(str(cc_output_file), str(final_target))
                print(f"Processing complete. File saved to: {final_target}")
                return final_target
            else:
                raise FileNotFoundError(
                    f"Expected CloudCompare output not found at: {cc_output_file}"
                )
        else:
            print(
                f"Processing complete. Look in {comp_path_obj.parent} for the generated file."
            )
            return cc_output_file

    def run_m3c2(
        self, overwrite=False
    ):  # NOTE possible to add core points (3rd cloud, subsampled reference) to speed up calculation if necessary
        """
        Executes CloudCompare M3C2 distance calculation headlessly. Need to first manually create m3c2_params.txt file first.
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

        ref_path_obj = Path(self.ref_path)
        cc_default_output = ref_path_obj.with_name(f"{ref_path_obj.stem}_M3C2.las")

        if self.output_dir:
            self.output_dir.mkdir(
                parents=True, exist_ok=True
            )  # Ensure the directory exists

            final_target = self.output_dir / cc_default_output.name

            # Move the file from the default location to the specified output directory
            if not overwrite:
                i = 1
                while Path(final_target).exists():
                    period = final_target.name.rfind(".")
                    bracket = final_target.name.find("(")

                    if bracket != -1:
                        name = f"{final_target.name[:bracket]}({i}){final_target.name[period:]}"
                    else:
                        name = f"{final_target.name[:period]}({i}){final_target.name[period:]}"

                    final_target = self.output_dir / name
                    i += 1

            if cc_default_output.exists():
                shutil.move(str(cc_default_output), str(final_target))
                print(f"Processing complete. File saved to: {final_target}")
                return final_target
            else:
                raise FileNotFoundError(
                    f"Expected CloudCompare output not found at: {cc_default_output}"
                )
        else:
            print(
                f"Processing complete. Look in {ref_path_obj.parent} for the generated file."
            )
            return cc_default_output

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

    def find_last_las(self) -> Path:
        patterns = [f"*_C2C_DIST_*.las", f"*M3C2*"]
        # matches = list(self.output_dir.glob(pattern))
        matches = []
        for p in patterns:
            matches.extend(self.output_dir.glob(p))

        last_file = max(matches, key=lambda p: p.stat().st_mtime)
        return Path(last_file)

    def run_cc(self, C2C=False, M3C2=False):
        if C2C:
            self.run_c2c()
        elif M3C2:
            self.run_m3c2()
        else:
            raise RuntimeError("Specify distance calculation method")

        last_las = self.find_last_las()

        pcd, dist = self.read_las_data(last_las)

        return pcd, dist

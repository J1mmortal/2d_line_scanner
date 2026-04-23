import subprocess
import os
import copy
import tempfile
import numpy as np
import open3d as o3d
import pyCloudCompare as cc


class CloudCompareC2CPipeline:
    """
    Uses CloudCompare CLI (via pyCloudCompareCLI) to compute C2C distances,
    then feeds results back into DamageDetector.

    Requirements:
        pip install pyCloudCompareCLI
        CloudCompare installed and on PATH (or pass cc_path explicitly)
    """

    def __init__(
        self,
        cc_path: str = "C:/Program Files/CloudCompare/CloudCompare.exe",
        output_dir: str = "../data/CC",
        no_timestamp: bool = True,
        spatial_subsample: float = None,
    ):
        self.cc_path = cc_path
        self.output_dir = output_dir or tempfile.mkdtemp(prefix="cc_c2c_")
        self.no_timestamp = no_timestamp
        self.spatial_subsample = spatial_subsample
        self._cli = cc.CloudCompareCLI()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _save_pcd(self, pcd: o3d.geometry.PointCloud, path: str):
        """Save an Open3D point cloud to PLY for CloudCompare."""
        o3d.io.write_point_cloud(path, pcd)

    def _find_output(self, prefix: str, suffix: str) -> str:
        """
        Find a CloudCompare output file in output_dir by prefix and suffix.
        Handles both timestamped and no-timestamp filenames.
        """
        candidates = [
            f
            for f in os.listdir(self.output_dir)
            if f.startswith(prefix) and f.endswith(suffix)
        ]
        if not candidates:
            raise FileNotFoundError(
                f"No CloudCompare output matching '{prefix}*{suffix}' "
                f"in {self.output_dir}.\n"
                f"Files present: {os.listdir(self.output_dir)}"
            )
        candidates.sort(
            key=lambda f: os.path.getmtime(os.path.join(self.output_dir, f))
        )
        return os.path.join(self.output_dir, candidates[-1])

    def _load_scalar_field(self, ply_path: str) -> np.ndarray:
        """
        Load a CloudCompare output PLY and extract the first scalar field
        (C2C distance) stored as vertex intensity.
        """
        pcd = o3d.io.read_point_cloud(ply_path)
        pts = np.asarray(pcd.points)

        # CloudCompare writes scalar fields as vertex colors (0-1 normalised)
        # when exporting PLY. Re-read with pyntcloud or directly parse the
        # scalar field from an ASC export (more reliable).
        raise RuntimeError(
            "Use _load_scalar_field_asc() for scalar field extraction — " "see below."
        )

    def _load_scalar_field_asc(self, asc_path: str) -> np.ndarray:
        """
        CloudCompare ASC output has format:
            X Y Z [R G B] ScalarField
        We take the last column as the C2C distance.
        """
        data = np.loadtxt(asc_path)
        return data[:, -1].astype(float)

    # ------------------------------------------------------------------
    # Main C2C pipeline
    # ------------------------------------------------------------------

    def compute_c2c(
        self,
        aligned_source: o3d.geometry.PointCloud,
        target: o3d.geometry.PointCloud,
    ) -> np.ndarray:
        """
        Export clouds to disk, run CloudCompare C2C, return distances array.
        The first cloud is the 'compared' cloud, second is 'reference'.
        """
        src_path = os.path.join(self.output_dir, "aligned_source.ply")
        tgt_path = os.path.join(self.output_dir, "target.ply")
        self._save_pcd(aligned_source, src_path)
        self._save_pcd(target, tgt_path)

        with self._cli.new_command() as cmd:
            cmd.silent()

            cmd.open(src_path)
            cmd.open(tgt_path)

            if self.spatial_subsample is not None:
                cmd.sub_sample(
                    cc.SUBSAMPLING_ALGORITHM.SPATIAL,
                    self.spatial_subsample,
                )

            cmd.c2c_dist()

            if self.no_timestamp:
                cmd.no_timestamp()

            cmd.cloud_export_format(
                cc.CLOUD_EXPORT_FORMAT.ASCII,
                extension="asc",
            )
            cmd.save_clouds()

        asc_path = self._find_output("aligned_source", "_C2C_DIST.asc")
        distances = self._load_scalar_field_asc(asc_path)
        return distances

    # ------------------------------------------------------------------
    # M3C2 pipeline (requires M3C2 plugin + parameter file)
    # ------------------------------------------------------------------

    def compute_m3c2(
        self,
        cloud1: o3d.geometry.PointCloud,
        cloud2: o3d.geometry.PointCloud,
        m3c2_params_file: str,
    ) -> np.ndarray:
        """
        Run CloudCompare M3C2 plugin.
        m3c2_params_file: path to a .txt params file exported from CC GUI.
        """
        c1_path = os.path.join(self.output_dir, "cloud1.ply")
        c2_path = os.path.join(self.output_dir, "cloud2.ply")
        self._save_pcd(cloud1, c1_path)
        self._save_pcd(cloud2, c2_path)

        with self._cli.new_command() as cmd:
            cmd.silent()
            cmd.open(c1_path)
            cmd.open(c2_path)
            cmd.m3c2(m3c2_params_file)

            if self.no_timestamp:
                cmd.no_timestamp()

            cmd.cloud_export_format(
                cc.CLOUD_EXPORT_FORMAT.ASCII,
                extension="asc",
            )
            cmd.save_clouds()

        asc_path = self._find_output("cloud1", "_M3C2.asc")
        distances = self._load_scalar_field_asc(asc_path)
        return distances

    # ------------------------------------------------------------------
    # SOR noise filter (optional preprocessing)
    # ------------------------------------------------------------------

    def sor_filter(
        self,
        pcd: o3d.geometry.PointCloud,
        n_neighbors: int = 6,
        sigma_multiplier: float = 1.0,
    ) -> o3d.geometry.PointCloud:
        """
        Apply CloudCompare SOR (Statistical Outlier Removal) and return
        the cleaned cloud as an Open3D PointCloud.
        """
        in_path = os.path.join(self.output_dir, "input_for_sor.ply")
        self._save_pcd(pcd, in_path)

        with self._cli.new_command() as cmd:
            cmd.silent()
            cmd.open(in_path)
            cmd.sor(n_neighbors, sigma_multiplier)

            if self.no_timestamp:
                cmd.no_timestamp()

            cmd.cloud_export_format(cc.CLOUD_EXPORT_FORMAT.PLY)
            cmd.save_clouds()

        out_path = self._find_output("input_for_sor", "_SOR.ply")
        return o3d.io.read_point_cloud(out_path)


# CC = r"C:\Program Files\CloudCompare\CloudCompare.exe"

src_path = "../data/Aligned_block.ply"
tgt_path = "../data/Original_block.ply"

ccl = CloudCompareC2CPipeline()

from damage_detection import DamageDetector
from global_reg import Registration

reg = Registration(course_voxel=3, voxel_size=3)
det = DamageDetector()

f1 = "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block.stl"
f2 = "C:/Users/fvsch/OneDrive/Desktop/TUDelft/Y3/BEP/Reg_block_tripledented.stl"

tgt, src = reg.poisson_convert(f1), reg.poisson_convert(f2)

icp, _ = reg.register(src, tgt)

aligned_src = copy.deepcopy(src)
aligned_src.transform(icp.transformation)

distances = ccl.compute_c2c(aligned_source=aligned_src, target=tgt)
print(distances)

det.visualise_colourmap(aligned_src, distances)

# result = subprocess.run(
#     [
#         CC,
#         "-VERBOSITY",
#         "3",
#         "-SILENT",
#         "-O",
#         src_path,
#         "-O",
#         tgt_path,
#         "-C2C_DIST",
#         "-MODEL",
#         "LS",
#         "SPHERE",
#         "3",
#     ],
#     capture_output=True,
#     text=True,
#     shell=False,
# )
# print("returncode:", result.returncode)
# print("stdout:", result.stdout)
# print("stderr:", result.stderr)

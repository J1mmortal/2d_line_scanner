import subprocess
from pathlib import Path


def run_c2c_pipeline(comp_path: Path, ref_path: Path, cc_exe: Path):
    """
    Executes CloudCompare C2C distance calculation headlessly.
    Returns the path to the newly generated file.
    """
    # Double-check paths before passing to C++ binary to prevent silent failures
    for path in [comp_path, ref_path, cc_exe]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    # Build the CLI command. Order is strictly enforced.
    cmd = [
        str(cc_exe),
        "-SILENT",  # Suppress the GUI
        "-C_EXPORT_FMT",
        "LAS",  # Output format. LAS/BIN preserve scalar fields.
        "-O",
        str(comp_path),  # File 1: The scan being inspected
        "-O",
        str(ref_path),  # File 2: The pristine reference
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
    print(f"Processing complete. Look in {comp_path.parent} for the generated file.")


def run_m3c2_pipeline(ref_path: Path, comp_path: Path, params_path: Path, cc_exe: Path):
    """
    Executes CloudCompare M3C2 distance calculation headlessly.
    """
    # Verify all components exist to prevent silent failures
    for path in [ref_path, comp_path, params_path, cc_exe]:
        if not path.exists():
            raise FileNotFoundError(f"Missing file: {path}")

    # Build the CLI command.
    # The order of -O arguments must match the indices expected by your params_path file.
    cmd = [
        str(cc_exe),
        "-SILENT",
        "-C_EXPORT_FMT",
        "LAS",
        "-O",
        str(ref_path),
        "-O",
        str(comp_path),
        "-M3C2",
        str(params_path),
    ]

    try:
        subprocess.run(cmd, capture_output=True, text=True, check=True)

    except subprocess.CalledProcessError as e:
        print(f"CloudCompare Exit Code: {e.returncode}")
        print(f"STDOUT:\n{e.stdout}")
        print(f"STDERR:\n{e.stderr}")
        raise RuntimeError("CloudCompare M3C2 pipeline failed. Review logs.")

    # CC saves the output in the directory of the compared file
    print(f"Processing complete. Check {comp_path.parent} for the generated file.")


# ---------------------------------------------------------
# Execution Example
# ---------------------------------------------------------
if __name__ == "__main__":
    # Assuming standard Windows installation path
    cc_executable = Path(r"C:\Program Files\CloudCompare\CloudCompare.exe")

    reference_scan = Path(
        r"C:\Users\fvsch\OneDrive\Documents\Code\TUDelft\Y3\BEP\2d_line_scanner\data\CC\TGT.ply"
    )
    compared_scan = Path(
        r"C:\Users\fvsch\OneDrive\Documents\Code\TUDelft\Y3\BEP\2d_line_scanner\data\CC\SRC.ply"
    )
    m3c2_parameters = Path(r"..\data\m3c2_params1.txt")

    # run_c2c_pipeline(compared_scan, reference_scan, cc_executable)
    run_m3c2_pipeline(reference_scan, compared_scan, m3c2_parameters, cc_executable)

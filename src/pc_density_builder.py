from pathlib import Path
import numpy as np
import trimesh


def triangle_areas_and_centroids(vertices, faces):
    tri = vertices[faces]                         # (F, 3, 3)
    v0 = tri[:, 0]
    v1 = tri[:, 1]
    v2 = tri[:, 2]

    cross = np.cross(v1 - v0, v2 - v0)
    areas = 0.5 * np.linalg.norm(cross, axis=1)
    centroids = (v0 + v1 + v2) / 3.0
    normals = cross / np.maximum(np.linalg.norm(cross, axis=1, keepdims=True), 1e-12)
    return areas, centroids, normals


def normalize_to_01(x):
    x = np.asarray(x, dtype=float)
    xmin, xmax = np.min(x), np.max(x)
    if xmax - xmin < 1e-12:
        return np.zeros_like(x)
    return (x - xmin) / (xmax - xmin)


def build_density_field(
    centroids,
    mode="smooth",
    axis=2,
    random_strength=1.0,
    smoothness=4.0,
    abrupt_threshold=0.5,
    low_density=0.2,
    high_density=3.0,
    seed=42):
    """
    Returns per-face density values (not yet multiplied by triangle area).

    modes:
      - 'uniform'
      - 'random'
      - 'smooth'
      - 'abrupt'
      - 'smooth_random'
      - 'abrupt_random'
    """
    rng = np.random.default_rng(seed)

    coord = normalize_to_01(centroids[:, axis])

    if mode == "uniform":
        density = np.ones(len(centroids))

    elif mode == "random":
        density = 1.0 + random_strength * rng.random(len(centroids))

    elif mode == "smooth":
        # Smooth variation along chosen axis using a sigmoid
        density = low_density + (high_density - low_density) / (
            1.0 + np.exp(-smoothness * (coord - 0.5))
        )

    elif mode == "abrupt":
        density = np.where(coord < abrupt_threshold, low_density, high_density)

    elif mode == "smooth_random":
        smooth = low_density + (high_density - low_density) / (
            1.0 + np.exp(-smoothness * (coord - 0.5))
        )
        rand = 1.0 + random_strength * (rng.random(len(centroids)) - 0.5)
        density = smooth * np.clip(rand, 0.1, None)

    elif mode == "abrupt_random":
        abrupt = np.where(coord < abrupt_threshold, low_density, high_density)
        rand = 1.0 + random_strength * (rng.random(len(centroids)) - 0.5)
        density = abrupt * np.clip(rand, 0.1, None)

    else:
        raise ValueError(f"Unknown mode: {mode}")

    return np.clip(density, 1e-8, None)


def sample_points_on_faces(vertices, faces, face_indices, seed=42):
    rng = np.random.default_rng(seed)
    tri = vertices[faces[face_indices]]   # (N, 3, 3)

    r1 = rng.random(len(face_indices))
    r2 = rng.random(len(face_indices))

    sqrt_r1 = np.sqrt(r1)
    u = 1.0 - sqrt_r1
    v = sqrt_r1 * (1.0 - r2)
    w = sqrt_r1 * r2

    points = (
        tri[:, 0] * u[:, None] +
        tri[:, 1] * v[:, None] +
        tri[:, 2] * w[:, None]
    )
    return points


def sample_nonuniform_point_cloud(
    stl_path,
    n_points=20000,
    mode="smooth",
    axis=2,
    random_strength=1.0,
    smoothness=6.0,
    abrupt_threshold=0.5,
    low_density=0.2,
    high_density=3.0,
    seed=42,
    export_ply=True,
    export_xyz=False,
    export_npy=False):
    
    stl_path = Path(stl_path)
    mesh = trimesh.load_mesh(stl_path)

    if not isinstance(mesh, trimesh.Trimesh):
        mesh = mesh.dump(concatenate=True)

    vertices = np.asarray(mesh.vertices)
    faces = np.asarray(mesh.faces)

    areas, centroids, normals = triangle_areas_and_centroids(vertices, faces)

    density = build_density_field(
        centroids,
        mode=mode,
        axis=axis,
        random_strength=random_strength,
        smoothness=smoothness,
        abrupt_threshold=abrupt_threshold,
        low_density=low_density,
        high_density=high_density,
        seed=seed
    )

    weights = areas * density
    weights = weights / np.sum(weights)

    rng = np.random.default_rng(seed)
    face_indices = rng.choice(len(faces), size=n_points, p=weights)
    points = sample_points_on_faces(vertices, faces, face_indices, seed=seed + 1)

    stem = stl_path.stem
    out_dir = stl_path.parent

    ply_path = out_dir / f"{stem}_{mode}_{n_points}.ply"
    xyz_path = out_dir / f"{stem}_{mode}_{n_points}.xyz"
    npy_path = out_dir / f"{stem}_{mode}_{n_points}.npy"

    if export_ply:
        pc = trimesh.points.PointCloud(points)
        pc.export(ply_path)

    if export_xyz:
        np.savetxt(xyz_path, points, fmt="%.8f")

    if export_npy:
        np.save(npy_path, points)

    return {
        "points": points,
        "face_indices": face_indices,
        "face_density": density,
        "face_weights": weights,
        "mesh": mesh,
        "ply_path": ply_path if export_ply else None,
        "xyz_path": xyz_path if export_xyz else None,
        "npy_path": npy_path if export_npy else None,
    }


if __name__ == "__main__":
    result = sample_nonuniform_point_cloud(
        stl_path=r"data/Reg_block_tripledented.STL",
        n_points=50000,
        # try: uniform, random, smooth, abrupt, smooth_random, abrupt_random
        mode="abrupt",
        export_ply=True,
        export_xyz=False,
        export_npy=False
    )
    print("PLY saved to:", result["ply_path"])
    print("Generated:", result["points"].shape[0], "points")
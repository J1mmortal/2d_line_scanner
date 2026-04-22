import numpy as np
from types import SimpleNamespace
import open3d as o3d
import copy

class DARERegistration:
    def __init__(self, k_neighbors=20, max_iter=20, corr_radius=0.1, tol=1e-5):
        self.k_neighbors = k_neighbors
        self.max_iter = max_iter
        self.corr_radius = corr_radius
        self.tol = tol

    def compute_density_weights(self, pts, kdtree, k=20):
        d = []
        for p in pts:
            _, idx, dist2 = kdtree.search_knn_vector_3d(p, k)
            if len(dist2) < 2:
                d.append(1.0)
                continue
            mean_d = np.sqrt(np.mean(dist2[1:]))
            d.append(mean_d)
        d = np.asarray(d)
        rho = 1.0 / (d + 1e-8)
        w = 1.0 / (rho + 1e-8)
        w = np.clip(w, np.percentile(w, 5), np.percentile(w, 95))
        w = w / (w.mean() + 1e-8)
        return w

    def weighted_rigid_transform(self, X, Y, W):
        W = W / (np.sum(W) + 1e-12)
        mx = np.sum(X * W[:, None], axis=0)
        my = np.sum(Y * W[:, None], axis=0)
        Xc = X - mx
        Yc = Y - my
        H = (Xc * W[:, None]).T @ Yc
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1
            R = Vt.T @ U.T
        t = my - R @ mx
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = t
        return T

    def register(self, source_pcd, target_pcd, init_transform):
        src = copy.deepcopy(source_pcd)
        tgt = copy.deepcopy(target_pcd)
        src.transform(init_transform)

        tgt_pts = np.asarray(tgt.points)
        tgt_tree = o3d.geometry.KDTreeFlann(tgt)
        T_total = init_transform.copy()

        for _ in range(self.max_iter):
            src_pts = np.asarray(src.points)
            src_tree = o3d.geometry.KDTreeFlann(src)

            ws = self.compute_density_weights(src_pts, src_tree, self.k_neighbors)
            wt = self.compute_density_weights(tgt_pts, tgt_tree, self.k_neighbors)

            X, Y, W = [], [], []
            for i, p in enumerate(src_pts):
                n, idx, dist2 = tgt_tree.search_hybrid_vector_3d(p, self.corr_radius, 1)
                if n < 1:
                    continue
                j = idx[0]
                X.append(p)
                Y.append(tgt_pts[j])
                W.append(ws[i] * wt[j])

            if len(X) < 6:
                break

            X = np.asarray(X)
            Y = np.asarray(Y)
            W = np.asarray(W)

            dT = self.weighted_rigid_transform(X, Y, W)
            src.transform(dT)
            T_total = dT @ T_total

            step = np.linalg.norm(dT[:3, 3]) + np.linalg.norm(dT[:3, :3] - np.eye(3))
            if step < self.tol:
                break

        return SimpleNamespace(
            transformation=T_total,
            correspondences=len(X) if len(X) else 0,
            debug={"iterations": self.max_iter}
        )
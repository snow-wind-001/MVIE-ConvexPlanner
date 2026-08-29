import numpy as np
import time
from scipy.optimize import minimize
from ..geometry import Ellipsoid

class MVIE_SOCP:
    """使用二阶锥规划(SOCP)方法计算最大体积内接椭球 (MVIE)"""
    def __init__(self, dimension=3):
        self.dim = dimension
        self.max_iterations = 200
        self.eps = 1e-8  # 减小精度以增强收敛
        self.condition_threshold = 1e6  # 降低条件数阈值以避免病态矩阵
    
    def compute(self, polytope):
        halfspaces = polytope.get_halfspaces()
        if halfspaces is None or halfspaces.shape[0] < self.dim + 1:
            ip = polytope.get_interior_point()
            interior_point = np.array(ip, dtype=float) if ip is not None else np.zeros(self.dim)
            return Ellipsoid(interior_point, np.eye(self.dim) * 0.1)
        
        A = halfspaces[:, :-1]
        b = -halfspaces[:, -1]
        
        norms = np.linalg.norm(A, axis=1)
        valid = norms > 1e-10
        A = A[valid] / norms[valid][:, np.newaxis]
        b = b[valid] / norms[valid]
        
        ip = polytope.get_interior_point()
        center = np.array(ip, dtype=float) if ip is not None else np.zeros(self.dim)
        
        slacks = b - A @ center
        if np.any(slacks <= 1e-10):
            return Ellipsoid(center, np.eye(self.dim) * 1e-4)

        # With normalized halfspaces, an isotropic ball of radius min(slack)
        # is feasible.  Starting strictly inside avoids SLSQP boundary issues.
        initial_radius = max(float(np.min(slacks)) * 0.8, 1e-5)
        initial_factor = np.eye(self.dim) * initial_radius
        candidates = [(initial_factor, center.copy())]

        try:
            factor, center_opt, result = self._solve_logdet_slsqp(
                A, b, center, initial_radius
            )
            self.last_result = result
            if factor is not None:
                candidates.append((factor, center_opt))
        except Exception as e:
            print(f"  MVIE log-det优化失败: {e}")

        best_ellipsoid = None
        best_volume = -np.inf
        for factor, center_opt in candidates:
            Q = factor @ factor.T
            if not self._is_valid_matrix(Q):
                continue
            # For E={c+Lu: ||u||<=1}, every halfspace must satisfy
            # a_i*c + ||L^T*a_i|| <= b_i.
            residual = A @ center_opt + np.linalg.norm(A @ factor, axis=1) - b
            if np.max(residual) > 1e-6:
                continue
            ellipsoid = Ellipsoid(center_opt, Q)
            volume = ellipsoid.volume()
            if np.isfinite(volume) and volume > best_volume:
                best_volume = volume
                best_ellipsoid = ellipsoid

        return best_ellipsoid or Ellipsoid(center, np.eye(self.dim) * 1e-4)

    def _solve_logdet_slsqp(self, A, b, center_init, radius_init):
        """Solve the convex MVIE log-det program with a Cholesky factor.

        The optimized ellipsoid is ``c + L u`` for ``||u|| <= 1``.  Positive
        diagonal entries of lower-triangular ``L`` are represented in log
        space, so positive definiteness is maintained throughout the solve.
        """
        lower_indices = [
            (row, col)
            for row in range(self.dim)
            for col in range(row + 1)
        ]
        factor_variables = []
        for row, col in lower_indices:
            factor_variables.append(np.log(radius_init) if row == col else 0.0)
        x0 = np.r_[center_init, factor_variables]

        max_slack = max(float(np.max(b - A @ center_init)), 1.0)
        factor_bound = max_slack * 10.0
        bounds = [(None, None)] * self.dim
        for row, col in lower_indices:
            if row == col:
                bounds.append((np.log(1e-8), np.log(factor_bound)))
            else:
                bounds.append((-factor_bound, factor_bound))

        def decode(values):
            center = values[: self.dim]
            factor = np.zeros((self.dim, self.dim))
            for value, (row, col) in zip(values[self.dim :], lower_indices):
                factor[row, col] = np.exp(value) if row == col else value
            return center, factor

        diagonal_positions = [
            self.dim + index
            for index, (row, col) in enumerate(lower_indices)
            if row == col
        ]

        def objective(values):
            # log(det(L)) is the log-volume up to an additive constant.
            return -float(np.sum(values[diagonal_positions]))

        def containment(values):
            center, factor = decode(values)
            support = np.linalg.norm(A @ factor, axis=1)
            return b - A @ center - support

        result = minimize(
            objective,
            x0,
            method="SLSQP",
            bounds=bounds,
            constraints=[{"type": "ineq", "fun": containment}],
            options={"maxiter": self.max_iterations, "ftol": 1e-10},
        )
        center, factor = decode(result.x)
        if np.min(containment(result.x)) < -1e-6:
            return None, center, result
        return factor, center, result
    
    def _solve_affine_scaling(self, A, b, center_init, max_iter=50, tol=1e-6):
        m, n = A.shape
        E = np.eye(n)
        center = center_init.copy()
        prev_log_vol = -np.inf
        for iter_idx in range(max_iter):
            slacks = b - A @ center
            if np.any(slacks <= 1e-12):
                break
            w = 1.0 / slacks
            H = (A.T * (w**2)) @ A + 1e-8 * np.eye(n)
            g = A.T @ w
            try:
                H_inv = np.linalg.inv(H)
                center -= 0.5 * (H_inv @ g)
                eigvals_h, eigvecs_h = np.linalg.eigh(H_inv)
                eigvals_h = np.maximum(eigvals_h, 1e-10)
                E = eigvecs_h @ np.diag(np.sqrt(eigvals_h))
                log_vol = 0.5 * np.sum(np.log(eigvals_h))
                if abs(log_vol - prev_log_vol) < tol and iter_idx > 3:
                    break
                prev_log_vol = log_vol
            except np.linalg.LinAlgError:
                break
        return E, center
    
    def _solve_khachiyan(self, A, b, center_init, tol=1e-6, max_iter=50):
        m, n = A.shape
        P = np.eye(n)
        center = center_init.copy()
        E = np.eye(n)
        for it in range(max_iter):
            denom_sq = np.array([float(A[i].T @ P @ A[i]) + 1e-8 for i in range(m)])
            safe_sqrt = np.sqrt(np.maximum(denom_sq, 1e-10))
            distances = np.where(b > 0, b / safe_sqrt, -1.0)
            min_dist = np.min(distances)
            if min_dist > 1.0 - tol:
                break
            j = np.argmin(distances)
            aj = A[j].reshape(-1, 1)
            sigma = (1.0 - min_dist) / (n + 1)
            P_aj = P @ aj
            denominator = float(aj.T @ P_aj) + 1e-8
            if denominator <= 0:
                break
            P = P - sigma * (P_aj @ P_aj.T) / denominator
            eigvals, eigvecs = np.linalg.eigh(P)
            eigvals = np.maximum(eigvals, 1e-6 * np.max(eigvals))
            E = eigvecs @ np.diag(1.0 / np.sqrt(eigvals)) @ eigvecs.T
        slacks = b - A @ center
        if np.all(slacks > 1e-10):
            for _ in range(20):
                s = b - A @ center
                if np.any(s <= 1e-10):
                    break
                w = 1.0 / s
                H_c = (A.T * (w**2)) @ A + 1e-8 * np.eye(n)
                g_c = A.T @ w
                try:
                    center -= 0.3 * np.linalg.solve(H_c, g_c)
                except np.linalg.LinAlgError:
                    break
        return E, center
    
    def _is_valid_matrix(self, Q):
        if np.any(np.isnan(Q)) or np.any(np.isinf(Q)):
            return False
        if not np.allclose(Q, Q.T, atol=1e-8):
            return False
        eigvals = np.linalg.eigvalsh(Q)
        if np.any(eigvals <= 0):
            return False
        cond = np.max(eigvals) / np.min(eigvals)
        return cond < self.condition_threshold

import numpy as np
from scipy.spatial import ConvexHull
from scipy.optimize import linprog
import warnings

class ConvexPolytope:
    def __init__(self, halfspaces=None, points=None):
        """
        创建一个凸多胞体
        可以使用半空间表示或点表示
        参数:
            halfspaces: 半空间表示，每行为 [a_1, a_2, ..., a_n, b]，表示 a·x + b <= 0
            points: 点表示，凸包的顶点
        """
        self.halfspaces = np.array(halfspaces) if halfspaces is not None else None
        self.points = np.array(points) if points is not None else None
        self.dimension = (halfspaces.shape[1] - 1) if halfspaces is not None else \
                        (points.shape[1] if points is not None else 0)
        self.interior_point = None
        self.vertices = None
        
    def get_vertices(self):
        """获取多胞体的顶点表示"""
        if self.points is not None:
            return self.points
        elif self.halfspaces is not None:
            if self.vertices is None:
                self.vertices = self.compute_vertices_from_halfspaces()
            return self.vertices
        else:
            print("警告: 没有点或半空间数据，无法获取顶点")
            return None
            
    def _sample_boundary_points(self, num_samples=1000):
        """
        在多胞体边界上均匀采样点
        参数:
            num_samples: 采样点数量
        返回:
            边界点列表
        """
        if self.points is None and self.halfspaces is None:
            return None
            
        interior_point = self.get_interior_point()
        if interior_point is None:
            print("无法找到内点，边界采样失败")
            return None
            
        indices = np.arange(0, num_samples, dtype=float) + 0.5
        phi = np.arccos(1 - 2*indices/num_samples)
        theta = np.pi * (1 + 5**0.5) * indices
        
        x = np.cos(theta) * np.sin(phi)
        y = np.sin(theta) * np.sin(phi)
        z = np.cos(phi)
        
        directions = np.vstack((x, y, z)).T
        
        if self.dimension != 3:
            if self.dimension == 2:
                directions = directions[:, :2]
            else:
                directions = np.random.randn(num_samples, self.dimension)
                norms = np.linalg.norm(directions, axis=1)
                directions = directions / norms[:, np.newaxis]
        
        boundary_points = []
        
        if self.halfspaces is not None:
            for direction in directions:
                intersections = []
                for i, hs in enumerate(self.halfspaces):
                    a = hs[:-1]
                    b = hs[-1]
                    denum = np.dot(direction, a)
                    if abs(denum) < 1e-10:
                        continue
                    t = (b - np.dot(interior_point, a)) / denum
                    if t > 0:
                        intersections.append((t, i))
                if intersections:
                    t_min = min(intersections)[0]
                    point = interior_point + t_min * direction
                    boundary_points.append(point)
        
        elif self.points is not None:
            try:
                hull = ConvexHull(self.points)
                vertices = self.points[hull.vertices]
                for direction in directions:
                    projections = np.dot(vertices, direction)
                    max_idx = np.argmax(projections)
                    boundary_points.append(vertices[max_idx])
            except:
                for direction in directions:
                    max_dist = -float('inf')
                    farthest_point = None
                    for point in self.points:
                        vec = point - interior_point
                        proj = np.dot(vec, direction)
                        if proj > max_dist:
                            max_dist = proj
                            farthest_point = point
                    if farthest_point is not None:
                        boundary_points.append(farthest_point)
        
        return np.array(boundary_points) if boundary_points else None
    
    def get_halfspaces(self):
        """获取多胞体的半空间表示"""
        if self.halfspaces is not None:
            return self.halfspaces
        elif self.points is not None:
            try:
                hull = ConvexHull(self.points)
                halfspaces = np.zeros((len(hull.equations), self.dimension + 1))
                halfspaces[:,:-1] = hull.equations[:,:-1]
                halfspaces[:,-1] = hull.equations[:,-1]
                return halfspaces
            except Exception as e:
                print(f"计算半空间表示出错: {e}")
        return None
    
    def get_interior_point(self):
        """
        获取多胞体内部的一个点
        使用多种方法尝试查找内点
        返回:
            内点坐标，如果找不到则返回None
        """
        if self.interior_point is not None:
            return self.interior_point
            
        methods = [
            self._get_interior_chebyshev,
            self._get_interior_lp,
            self._get_interior_from_vertices,
            self._get_interior_random
        ]
        
        for method in methods:
            try:
                point = method()
                if point is not None and self.contains(point):
                    self.interior_point = point
                    return point
            except Exception:
                continue
        
        warnings.warn("未能找到多胞体内点，返回默认点")
        default_point = np.zeros(self.dimension)
        self.interior_point = default_point
        return default_point
    
    def _get_interior_from_vertices(self):
        """使用已有顶点平均作为内点（不触发顶点计算以避免循环递归）"""
        if self.points is not None and len(self.points) > 0:
            return np.mean(self.points, axis=0)
        if self.vertices is not None and len(self.vertices) > 0:
            return np.mean(self.vertices, axis=0)
        return None
    
    def _get_interior_chebyshev(self):
        """使用 linprog 计算 Chebyshev 中心"""
        if self.halfspaces is None:
            return None
            
        A = self.halfspaces[:, :-1]
        b = -self.halfspaces[:, -1]
        
        c = np.concatenate([[-1.0], np.zeros(self.dimension)])
        A_ub = np.hstack((np.linalg.norm(A, axis=1)[:, np.newaxis], A))
        b_ub = b
        
        bounds = [(0, None)] + [(None, None)] * self.dimension
        res = linprog(c, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method='highs')
        
        if res.success:
            return res.x[1:]
        return None
    
    def _get_interior_lp(self):
        """使用简单 LP 求解内点"""
        if self.halfspaces is None:
            return None
            
        A = self.halfspaces[:, :-1]
        b = -self.halfspaces[:, -1]
        c = np.zeros(self.dimension)
        res = linprog(c, A_ub=A, b_ub=b, method='highs')
        if res.success:
            return res.x
        return None
    
    def _get_interior_random(self):
        """随机采样内点"""
        if self.halfspaces is None:
            return None
            
        for _ in range(100):
            point = np.random.uniform(-10, 10, self.dimension)
            if self.contains(point):
                return point
        return None
    
    def contains(self, point):
        """检查点是否在多胞体内"""
        if self.halfspaces is None:
            return False
        return np.all(self.halfspaces[:, :-1] @ point + self.halfspaces[:, -1] <= 1e-6)
    
    def compute_vertices_from_halfspaces(self):
        """计算多胞体的顶点"""
        interior_point = self.get_interior_point()
        if interior_point is None:
            return None
            
        methods = [self._compute_vertices_ray_casting, self._compute_vertices_from_face_intersections]
        
        for method in methods:
            vertices = method(interior_point)
            if vertices is not None and len(vertices) >= self.dimension + 1:
                return vertices
        
        return None
    
    def _compute_vertices_ray_casting(self, interior_point):
        """使用射线投射法计算顶点"""
        if self.halfspaces is None:
            return None
            
        num_directions = min(5000, 100 * self.dimension)
        directions = []
        
        if self.dimension == 3:
            golden_ratio = (1 + 5 ** 0.5) / 2
            indices = np.arange(num_directions)
            theta = 2 * np.pi * indices / golden_ratio
            phi = np.arccos(1 - 2 * (indices + 0.5) / num_directions)
            x = np.sin(phi) * np.cos(theta)
            y = np.sin(phi) * np.sin(theta)
            z = np.cos(phi)
            directions = np.vstack([x, y, z]).T
        elif self.dimension == 2:
            angles = np.linspace(0, 2 * np.pi, num_directions, endpoint=False)
            directions = np.column_stack([np.cos(angles), np.sin(angles)])
        else:
            while len(directions) < num_directions:
                direction = np.random.normal(size=self.dimension)
                norm = np.linalg.norm(direction)
                if norm > 1e-10:
                    directions.append(direction / norm)
            directions = np.array(directions)
        
        intersection_points = []
        A = self.halfspaces[:, :-1]
        b = self.halfspaces[:, -1]
        
        for direction in directions:
            numerator = -(A @ interior_point + b)
            denominator = A @ direction
            valid_indices = (denominator < -1e-10) | (denominator > 1e-10)
            
            if np.sum(valid_indices) > 0:
                t_values = np.zeros_like(numerator)
                t_values[valid_indices] = numerator[valid_indices] / denominator[valid_indices]
                positive_t = t_values[t_values > 1e-10]
                if len(positive_t) > 0:
                    min_t = np.min(positive_t)
                    intersection_point = interior_point + min_t * direction
                    if self._is_vertex(intersection_point):
                        intersection_points.append(intersection_point)
        
        if len(intersection_points) > 0:
            from scipy.spatial import cKDTree
            tree = cKDTree(intersection_points)
            unique_indices = []
            tolerance = max(1e-6, 1e-4 * np.max([np.linalg.norm(p) for p in intersection_points]))
            for i, point in enumerate(intersection_points):
                indices = tree.query_ball_point(point, tolerance)
                if i == min(indices):
                    unique_indices.append(i)
            return np.array([intersection_points[i] for i in unique_indices])
        
        return None
    
    def _compute_vertices_from_face_intersections(self, interior_point):
        """通过计算面交点来找到顶点"""
        from itertools import combinations
        vertices = []
        n = len(self.halfspaces)
        A = self.halfspaces[:, :-1]
        b = -self.halfspaces[:, -1]
        
        for face_indices in combinations(range(n), self.dimension):
            A_sub = A[list(face_indices)]
            b_sub = b[list(face_indices)]
            if np.linalg.matrix_rank(A_sub, tol=1e-6) == self.dimension:
                try:
                    vertex = np.linalg.solve(A_sub, b_sub)
                    if self.contains(vertex):
                        is_duplicate = False
                        for v in vertices:
                            if np.linalg.norm(vertex - v) < 1e-6:
                                is_duplicate = True
                                break
                        if not is_duplicate:
                            vertices.append(vertex)
                except:
                    continue
        
        return np.array(vertices) if vertices else None
    
    def _is_vertex(self, point):
        """检查点是否为多胞体的真实顶点"""
        on_face_indices = [i for i, hs in enumerate(self.halfspaces) if abs(np.dot(hs[:-1], point) + hs[-1]) < 1e-6]
        if len(on_face_indices) >= self.dimension:
            face_normals = self.halfspaces[on_face_indices, :-1]
            return np.linalg.matrix_rank(face_normals, tol=1e-6) >= self.dimension
        return False
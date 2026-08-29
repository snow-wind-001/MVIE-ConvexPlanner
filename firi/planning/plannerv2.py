import numpy as np
import time
import os
import pickle
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from .firi import FIRI
from .config import FIRIConfig
from .spherical_projection import (
    SphericalProjectionConfig,
    SphericalProjectionGuide,
)
from ..geometry import Ellipsoid

class FIRIPlanner:
    def __init__(self, obstacles, space_size, space_bounds=None):
        self.obstacles = obstacles
        self.space_size = space_size
        self.space_bounds = space_bounds
        self.dimension = len(space_size)
        self.config = FIRIConfig(space_size)
        self.config.update_adaptive_params(obstacle_count=len(obstacles))
        self.firi = FIRI(obstacles, self.dimension, space_bounds=space_bounds)
        self._build_obstacle_kdtree()
        self.safe_regions = []
        self.path_points = []
        self.path_collisions = []
        self.last_optimization_result = None
        self.last_optimization_feasible = False
        self.spherical_guide = None
        self.realtime_guide = None
        self.last_realtime_stats = {}
        self.safety_margin = float(self.config.safety_margin)
        self.set_safety_margin(self.safety_margin)

    def set_safety_margin(self, safety_margin):
        """Set the absolute obstacle clearance used by every planner stage."""
        safety_margin = float(safety_margin)
        if not np.isfinite(safety_margin) or safety_margin < 0.0:
            raise ValueError("safety_margin must be a finite non-negative value")
        self.safety_margin = safety_margin
        self.config.safety_margin = safety_margin
        self.firi.safety_margin = safety_margin
        projection_far = max(float(np.linalg.norm(self.space_size)), 1.0)
        self.spherical_guide = SphericalProjectionGuide(
            self.obstacles,
            config=SphericalProjectionConfig(
                width=41,
                height=25,
                far=projection_far,
                safety_radius=safety_margin,
                max_candidates=16,
            ),
        )
        self.realtime_guide = SphericalProjectionGuide(
            self.obstacles,
            config=SphericalProjectionConfig(
                # The reactive layer favors a small, bounded angular map.
                # Exact 3-D checks remain authoritative for every candidate.
                width=17,
                height=13,
                far=min(projection_far, 8.0),
                safety_radius=safety_margin,
                max_candidates=16,
            ),
        )
        
    def _build_obstacle_kdtree(self):
        vertices = []
        self.obstacle_radii = []
        sphere_centers = []
        sphere_radii = []
        cylinder_centers = []
        cylinder_radii = []
        cylinder_heights = []
        cuboid_centers = []
        cuboid_half_sizes = []
        capsule_starts = []
        capsule_ends = []
        capsule_radii = []
        self._analytic_obstacles_complete = True
        for obs in self.obstacles:
            try:
                center = np.array(obs.center)
                shape = getattr(obs, 'shape', 'sphere')
                num_samples = 30

                if shape == 'cuboid' and getattr(obs, 'size', None) is not None:
                    size = np.array(obs.size)
                    half_sizes = size / 2
                    corners_x = np.array([-1, 1, 1, -1, -1, 1, 1, -1]) * half_sizes[0]
                    corners_y = np.array([-1, -1, 1, 1, -1, -1, 1, 1]) * half_sizes[1]
                    corners_z = np.array([-1, -1, -1, -1, 1, 1, 1, 1]) * half_sizes[2]
                    vertices.extend(np.column_stack((corners_x, corners_y, corners_z)) + center)
                    self.obstacle_radii.append((center, half_sizes))
                    cuboid_centers.append(center)
                    cuboid_half_sizes.append(half_sizes)
                elif shape == 'cylinder' and getattr(obs, 'height', None) is not None:
                    radius = obs.radius if obs.radius is not None else 1.0
                    height = obs.height
                    theta = np.linspace(0, 2 * np.pi, num_samples)
                    x = center[0] + radius * np.cos(theta)
                    y = center[1] + radius * np.sin(theta)
                    z = np.linspace(center[2] - height / 2, center[2] + height / 2, num_samples)
                    vertices.extend(np.column_stack((x, y, z)))
                    self.obstacle_radii.append((center, radius, height))
                    cylinder_centers.append(center)
                    cylinder_radii.append(radius)
                    cylinder_heights.append(height)
                elif (
                    shape == 'capsule'
                    and getattr(obs, 'start', None) is not None
                    and getattr(obs, 'end', None) is not None
                ):
                    start = np.asarray(obs.start, dtype=float)
                    end = np.asarray(obs.end, dtype=float)
                    radius = float(obs.radius if obs.radius is not None else 1.0)
                    axis_samples = np.linspace(0.0, 1.0, 8)[:, None]
                    vertices.extend(start + axis_samples * (end - start))
                    # The slow FIRI seed-push stage uses this conservative
                    # bounding sphere.  Realtime collision checks below use
                    # the exact capsule geometry.
                    bound_radius = radius + np.linalg.norm(end - start) / 2.0
                    self.obstacle_radii.append((center, bound_radius))
                    capsule_starts.append(start)
                    capsule_ends.append(end)
                    capsule_radii.append(radius)
                elif shape == 'sphere':
                    radius = obs.radius if obs.radius is not None else 1.0
                    theta = np.linspace(0, 2 * np.pi, num_samples)
                    x = center[0] + radius * np.cos(theta)
                    y = center[1] + radius * np.sin(theta)
                    z = np.linspace(center[2] - radius, center[2] + radius, num_samples)
                    vertices.extend(np.column_stack((x, y, z)))
                    self.obstacle_radii.append((center, radius))
                    sphere_centers.append(center)
                    sphere_radii.append(radius)
                else:
                    self._analytic_obstacles_complete = False
            except Exception as e:
                print(f"构建KD树时处理障碍物出错: {e}")
                self._analytic_obstacles_complete = False
                continue
        self._sphere_centers = np.asarray(sphere_centers, dtype=float).reshape(-1, 3)
        self._sphere_radii = np.asarray(sphere_radii, dtype=float)
        self._cylinder_centers = np.asarray(
            cylinder_centers, dtype=float
        ).reshape(-1, 3)
        self._cylinder_radii = np.asarray(cylinder_radii, dtype=float)
        self._cylinder_heights = np.asarray(cylinder_heights, dtype=float)
        self._cuboid_centers = np.asarray(cuboid_centers, dtype=float).reshape(-1, 3)
        self._cuboid_half_sizes = np.asarray(
            cuboid_half_sizes, dtype=float
        ).reshape(-1, 3)
        self._capsule_starts = np.asarray(capsule_starts, dtype=float).reshape(-1, 3)
        self._capsule_ends = np.asarray(capsule_ends, dtype=float).reshape(-1, 3)
        self._capsule_radii = np.asarray(capsule_radii, dtype=float)
        if vertices:
            self.obstacle_tree = KDTree(vertices)
            print(f"已构建KD-Tree: {len(vertices)}个顶点")
        else:
            self.obstacle_tree = None
            print("警告: 无法构建障碍物KD树")
    
    def generate_safe_regions(self, start, goal, num_waypoints=6):
        self.safe_regions = []
        os.makedirs('temp', exist_ok=True)
        t_values = np.linspace(0, 1, num_waypoints+1)
        path_points = np.array([start * (1-t) + goal * t for t in t_values])
        pickle.dump(path_points, open('temp/adjusted_path.pkl', 'wb'))
        for i in range(len(path_points) - 1):
            p1 = path_points[i]
            p2 = path_points[i+1]
            mid_point = (p1 + p2) / 2
            direction = p2 - p1
            direction_norm = np.linalg.norm(direction)
            if direction_norm > 1e-10:
                direction = direction / direction_norm
            else:
                direction = np.random.randn(self.dimension)
                direction = direction / np.linalg.norm(direction)
            if self.dimension == 3:
                if abs(direction[0]) < abs(direction[1]):
                    normal1 = np.array([1, 0, 0])
                else:
                    normal1 = np.array([0, 1, 0])
                normal1 = np.cross(direction, normal1)
                normal1 = normal1 / np.linalg.norm(normal1)
                normal2 = np.cross(direction, normal1)
                normal2 = normal2 / np.linalg.norm(normal2)
            else:
                normal1 = np.array([-direction[1], direction[0]])
                normal2 = -normal1
            seed_points = [p1, mid_point, p2]
            print(f"为路径段 {i} 计算安全区域 (包含 {len(seed_points)} 个种子点)...")
            start_time = time.time()
            iterations = self.config.safety_iterations
            threshold = self.config.volume_threshold
            try:
                polytope, ellipsoid = self.firi.compute_safe_region(
                    seed_points, 
                    max_iterations=iterations, 
                    volume_threshold=threshold
                )
                self.safe_regions.append((polytope, ellipsoid))
                print(f"安全区域 {i} 椭球体体积: {ellipsoid.volume():.6f}, 计算时间: {time.time() - start_time:.2f}秒")
            except Exception as e:
                print(f"计算安全区域 {i} 出错: {e}")
                self.safe_regions.append(None)
        return self.safe_regions
    
    def generate_initial_waypoints(self, start, goal, num_waypoints=6, jitter_ratio=0.05):
        """
        生成更合理的初始路径点，扰动仅在主方向的垂直平面内，且扰动幅度在路径中间最大，两端为0
        jitter_ratio: 扰动占总路径长度的比例（如0.05表示5%）
        """
        waypoints = [start]
        direction = goal - start
        direction_norm = np.linalg.norm(direction)
        if direction_norm < 1e-8:
            raise ValueError("起点和终点重合")
        unit_dir = direction / direction_norm

        # 构造一个正交基
        if abs(unit_dir[0]) < 0.9:
            ortho = np.array([1, 0, 0])
        else:
            ortho = np.array([0, 1, 0])
        perp1 = np.cross(unit_dir, ortho)
        perp1 = perp1 / np.linalg.norm(perp1)
        perp2 = np.cross(unit_dir, perp1)
        perp2 = perp2 / np.linalg.norm(perp2)

        for i in range(1, num_waypoints):
            t = i / num_waypoints
            base_point = start * (1-t) + goal * t
            # 扰动幅度在路径中间最大，两端为0
            disturbance = np.sin(np.pi * t)
            # 随机在两个正交方向上扰动
            angle = np.random.uniform(0, 2 * np.pi)
            jitter_vec = np.cos(angle) * perp1 + np.sin(angle) * perp2
            jitter_length = disturbance * jitter_ratio * direction_norm
            waypoint = base_point + jitter_length * jitter_vec
            waypoints.append(waypoint)
        waypoints.append(goal)
        return np.array(waypoints)

    # =====================================================================
    # 伪代码 Steps 5-13: 迭代安全推离
    # =====================================================================
    def _safety_push(self, waypoints):
        """
        在 FIRI 之前将控制点推离障碍物，确保种子点位于安全空间。
        伪代码:
            for iter = 1 to 10:
                for x where d_min(x) < d_safe:
                    n̂ = (x - p_near) / ||x - p_near||
                    x = x + (d_safe - d_min(x) + δ) * n̂
        """
        d_safe = max(self.config.d_safe, self.safety_margin)
        delta = self.config.push_delta
        max_iter = self.config.push_iterations
        pushed = waypoints.copy()

        bounds_min = bounds_max = None
        if self.space_bounds is not None:
            bounds_min = np.array(self.space_bounds[0]) + 0.1
            bounds_max = np.array(self.space_bounds[1]) - 0.1

        for iteration in range(max_iter):
            any_pushed = False
            for i in range(1, len(pushed) - 1):
                if self.obstacle_tree is None:
                    continue
                dist_kd, idx_kd = self.obstacle_tree.query(pushed[i])

                min_dist = dist_kd
                p_near = self.obstacle_tree.data[idx_kd]
                for center, *params in self.obstacle_radii:
                    if len(params) == 1:
                        p = params[0]
                        if np.isscalar(p) or (isinstance(p, np.ndarray) and p.ndim == 0):
                            d = np.linalg.norm(pushed[i] - center) - float(p)
                        else:
                            d = float(np.max(np.abs(pushed[i] - center) - np.array(p)))
                    elif len(params) == 2:
                        r, h = float(params[0]), float(params[1])
                        d = max(np.linalg.norm(pushed[i][:2] - center[:2]) - r,
                                abs(pushed[i][2] - center[2]) - h / 2)
                    else:
                        continue
                    if d < min_dist:
                        min_dist = d
                        p_near = center

                if min_dist < d_safe:
                    direction = pushed[i] - p_near
                    d = np.linalg.norm(direction)
                    if d < 1e-8:
                        direction = np.random.randn(self.dimension)
                        d = np.linalg.norm(direction)
                    n_hat = direction / d
                    push_amount = d_safe - min_dist + delta
                    pushed[i] = pushed[i] + push_amount * n_hat
                    if bounds_min is not None:
                        pushed[i] = np.clip(pushed[i], bounds_min, bounds_max)
                    any_pushed = True

            if not any_pushed:
                break

        pushed_count = sum(
            1 for i in range(1, len(pushed) - 1)
            if np.linalg.norm(pushed[i] - waypoints[i]) > 1e-6
        )
        if pushed_count > 0:
            print(f"安全推离: {pushed_count} 个控制点被推离障碍物 ({iteration+1} 轮迭代)")
        return pushed

    def _projection_refine(self, waypoints, max_passes=2):
        """Insert projection-guided waypoints before corridor construction."""
        if self.spherical_guide is None or len(waypoints) < 2:
            return waypoints
        current = np.asarray(waypoints, dtype=float)
        for _ in range(max_passes):
            collisions = self.check_path_safety(current)
            if not collisions:
                break
            new_points = list(current)
            offset = 0
            inserted = 0
            for segment_index in collisions:
                index = segment_index + offset
                if index >= len(new_points) - 1:
                    continue
                p1 = np.asarray(new_points[index], dtype=float)
                p2 = np.asarray(new_points[index + 1], dtype=float)
                bypass = self.spherical_guide.find_bypass(
                    p1,
                    p2,
                    point_collision=self.check_point_collision,
                    segment_collision=self.check_segment_collision,
                    bounds=self.space_bounds,
                )
                if bypass is not None:
                    new_points.insert(index + 1, bypass)
                    offset += 1
                    inserted += 1
            if inserted == 0:
                break
            candidate = np.asarray(new_points, dtype=float)
            if len(self.check_path_safety(candidate)) < len(collisions):
                current = candidate
            else:
                break
        if len(current) != len(waypoints):
            print(
                f"球面投影引导: {len(waypoints)} -> {len(current)} 个控制点, "
                f"剩余碰撞段 {len(self.check_path_safety(current))}"
            )
        return current

    # =====================================================================
    # 伪代码 Steps 15-22: 根据修正后控制点计算安全走廊
    # =====================================================================
    def _compute_corridors(self, waypoints):
        """
        为每段路径计算安全区域 (FIRI + MVIE)，使用修正后的控制点作为种子。
        """
        corridors = []
        for i in range(len(waypoints) - 1):
            p1 = waypoints[i]
            p2 = waypoints[i + 1]
            mid = (p1 + p2) / 2
            seed_points = [p1, mid, p2]
            start_time = time.time()
            try:
                polytope, ellipsoid = self.firi.compute_safe_region(
                    seed_points,
                    max_iterations=self.config.safety_iterations,
                    volume_threshold=self.config.volume_threshold
                )
                corridors.append((polytope, ellipsoid))
                print(f"走廊 {i}: 体积={ellipsoid.volume():.4f}, 耗时={time.time()-start_time:.2f}s")
            except Exception as e:
                print(f"走廊 {i} 计算失败: {e}")
                corridors.append(None)
        self.safe_regions = corridors
        return corridors

    # =====================================================================
    # 伪代码 Step 25: 约束轨迹优化
    # =====================================================================
    def _optimize_trajectory(self, waypoints, corridors):
        """
        以安全走廊、加速度、jerk 为约束，优化控制点位置。
        min  Σ ||P_{i+2} - 2P_{i+1} + P_i||²   (平滑性)
        s.t. 走廊约束:  各段采样点在对应椭球体内
             加速度约束: ||二阶差分|| ≤ a_max
             jerk约束:  ||三阶差分|| ≤ jerk_max
        """
        n = len(waypoints)
        dim = self.dimension
        if n < 4:
            return waypoints

        start = waypoints[0].copy()
        goal = waypoints[-1].copy()
        n_inner = n - 2
        x0 = waypoints[1:-1].flatten().copy()

        a_max = self.config.a_max
        jerk_max = self.config.jerk_max

        corridor_data = []
        for k, corr in enumerate(corridors):
            if corr is None or k >= n - 1:
                continue
            polytope, _ = corr
            halfspaces = polytope.get_halfspaces()
            if halfspaces is None or len(halfspaces) == 0:
                continue
            A = np.asarray(halfspaces[:, :-1], dtype=float)
            offset = np.asarray(halfspaces[:, -1], dtype=float)
            valid = np.all(np.isfinite(A), axis=1) & np.isfinite(offset)
            if np.any(valid):
                corridor_data.append((k, A[valid], offset[valid]))

        def _all_pts(x):
            inner = x.reshape(n_inner, dim)
            return np.vstack([start, inner, goal])

        def objective(x):
            pts = _all_pts(x)
            cost = 0.0
            for i in range(len(pts) - 2):
                d2 = pts[i + 2] - 2 * pts[i + 1] + pts[i]
                cost += np.dot(d2, d2)
            return cost

        constraints = []

        for k, A_k, offset_k in corridor_data:
            # A convex segment lies in the polytope if both endpoints do.
            for endpoint_offset in (0, 1):
                def _corr(x, _k=k, _offset=endpoint_offset, _A=A_k, _b=offset_k):
                    pts = _all_pts(x)
                    point_index = _k + _offset
                    if point_index >= len(pts):
                        return np.ones(len(_b))
                    return -(_A @ pts[point_index] + _b)
                constraints.append({'type': 'ineq', 'fun': _corr})

        a_max_sq = a_max ** 2
        for i in range(n - 2):
            def _acc(x, _i=i, _a2=a_max_sq):
                pts = _all_pts(x)
                d2 = pts[_i + 2] - 2 * pts[_i + 1] + pts[_i]
                return _a2 - np.dot(d2, d2)
            constraints.append({'type': 'ineq', 'fun': _acc})

        jerk_max_sq = jerk_max ** 2
        for i in range(n - 3):
            def _jrk(x, _i=i, _j2=jerk_max_sq):
                pts = _all_pts(x)
                d3 = pts[_i + 3] - 3 * pts[_i + 2] + 3 * pts[_i + 1] - pts[_i]
                return _j2 - np.dot(d3, d3)
            constraints.append({'type': 'ineq', 'fun': _jrk})

        bounds = None
        if self.space_bounds is not None:
            lb = np.tile(np.array(self.space_bounds[0]) + 0.1, n_inner)
            ub = np.tile(np.array(self.space_bounds[1]) - 0.1, n_inner)
            bounds = list(zip(lb, ub))

        try:
            result = minimize(
                objective, x0,
                method='SLSQP',
                constraints=constraints,
                bounds=bounds,
                options={'maxiter': self.config.opt_max_iter, 'ftol': 1e-6}
            )
            self.last_optimization_result = result
            optimized = _all_pts(result.x)
            constraint_values = [
                np.min(np.atleast_1d(constraint['fun'](result.x)))
                for constraint in constraints
            ]
            self.last_optimization_feasible = bool(
                not constraint_values or min(constraint_values) >= -1e-5
            )
            if result.success:
                print(f"轨迹优化收敛 (iter={result.nit}, cost={result.fun:.4f})")
            else:
                print(f"轨迹优化未完全收敛: {result.message}")

            if not self.last_optimization_feasible:
                print("轨迹优化结果违反约束，使用投影引导路径")
                return waypoints

            opt_col = self.check_path_safety(optimized)
            orig_col = self.check_path_safety(waypoints)
            if len(opt_col) <= len(orig_col):
                return optimized
            else:
                print(f"优化后碰撞增加 ({len(orig_col)}->{len(opt_col)})，使用原始路径")
                return waypoints
        except Exception as e:
            print(f"轨迹优化异常: {e}，使用原始路径")
            return waypoints

    # =====================================================================
    # 路径后处理: 去重与简化
    # =====================================================================
    def _deduplicate_path(self, path, min_dist=0.1):
        """Remove near-duplicate consecutive points, always keeping start and goal."""
        if len(path) <= 2:
            return path
        result = [path[0]]
        for i in range(1, len(path) - 1):
            if np.linalg.norm(path[i] - result[-1]) > min_dist:
                result.append(path[i])
        result.append(path[-1])
        return np.array(result)

    def _simplify_path(self, path):
        """Greedily skip intermediate waypoints when a direct segment is collision-free."""
        if len(path) <= 3:
            return path
        result = [path[0]]
        i = 0
        while i < len(path) - 1:
            farthest = i + 1
            for j in range(len(path) - 1, i + 1, -1):
                if not self.check_segment_collision(path[i], path[j]):
                    farthest = j
                    break
            result.append(path[farthest])
            i = farthest
        return np.array(result)

    # =====================================================================
    # 主规划流程 (按伪代码 Algorithm 1 重构)
    # =====================================================================
    def plan_path(self, start, goal, initial_waypoints=None, smoothing=True,
                  max_replanning_attempts=10, safety_margin=1.0,
                  use_spherical_guidance=True):
        os.makedirs('temp', exist_ok=True)
        self.set_safety_margin(safety_margin)
        straight_length = np.linalg.norm(goal - start)

        # Step 3-4: 生成初始控制点 + 正弦扰动
        if initial_waypoints is None:
            num_waypoints = 6
            init_path = self.generate_initial_waypoints(
                start, goal, num_waypoints=num_waypoints, jitter_ratio=0.05)
        else:
            init_path = np.array(initial_waypoints)

        # Steps 5-13: 迭代安全推离
        pushed_path = self._safety_push(init_path)

        # Kimi方案的视锥->球面->平面通道图作为局部引导前端。
        if use_spherical_guidance:
            pushed_path = self._projection_refine(pushed_path)

        # Steps 15-22: 计算安全走廊
        corridors = self._compute_corridors(pushed_path)

        # Step 25: 约束轨迹优化
        optimized_path = self._optimize_trajectory(pushed_path, corridors)

        # Step 24/26: B-spline 平滑 + 安全验证
        final_path = optimized_path
        if smoothing:
            try:
                bspline_path = self.bspline_smooth(optimized_path, smoothing_factor=0.5)
                if not self.check_path_safety(bspline_path):
                    print("B-spline平滑成功且安全")
                    final_path = bspline_path
                else:
                    print("B-spline平滑后有碰撞，保留优化路径")
            except Exception as e:
                print(f"B-spline平滑出错: {e}")

        # 验证最终路径
        collisions = self.check_path_safety(final_path)

        # 如果仍有碰撞，用旧版启发式重规划作为 fallback
        if collisions:
            print(f"优化管线后仍有 {len(collisions)} 段碰撞，启用启发式重规划...")
            final_path = self._fallback_replan(
                pushed_path, start, goal, corridors, max_replanning_attempts, smoothing)
            collisions = self.check_path_safety(final_path)

        # 裁剪到边界
        if self.space_bounds is not None:
            final_path = np.clip(
                final_path, self.space_bounds[0], self.space_bounds[1])

        final_path = self._deduplicate_path(final_path)
        final_path = self._simplify_path(final_path)

        # Post-processing can change segment indices and collision state.
        collisions = self.check_path_safety(final_path)

        self.path_points = final_path
        self.path_collisions = collisions
        path_length = np.sum(np.linalg.norm(np.diff(final_path, axis=0), axis=1))
        print(f"最终路径点: {final_path}")
        print(f"最终路径长度: {path_length:.2f} (直线长度: {straight_length:.2f})")

        if collisions:
            print(f"警告: 仍有 {len(collisions)} 处碰撞")
        else:
            print("路径安全，无碰撞")

        try:
            with open('temp/path_points.pkl', 'wb') as path_file:
                pickle.dump(final_path, path_file)
            with open('temp/adjusted_path.pkl', 'wb') as adjusted_file:
                pickle.dump(final_path, adjusted_file)
        except Exception:
            pass
        return final_path

    def plan_realtime(
        self,
        start,
        goal,
        reference_path=None,
        safety_margin=None,
        time_budget=0.020,
        max_repairs=1,
    ):
        """Fast local update intended for a 50 Hz control loop.

        FIRI/MVIE corridor construction remains a low-frequency global task.
        This method validates a cached reference path and performs at most a
        bounded number of low-resolution spherical-map repairs.  It returns
        ``None`` instead of entering an unbounded fallback when no verified
        solution fits the real-time budget.
        """
        started = time.perf_counter()
        if safety_margin is not None and not np.isclose(
            float(safety_margin), self.safety_margin
        ):
            self.set_safety_margin(safety_margin)
        start = np.asarray(start, dtype=float)
        goal = np.asarray(goal, dtype=float)
        if reference_path is None:
            if len(self.path_points) >= 2:
                path = np.asarray(self.path_points, dtype=float).copy()
            else:
                path = np.vstack([start, goal])
        else:
            path = np.asarray(reference_path, dtype=float).copy()
        if path.ndim != 2 or path.shape[1] != self.dimension or len(path) < 2:
            raise ValueError("reference_path must have shape (N, dimension), N >= 2")
        path[0] = start
        path[-1] = goal
        if self.space_bounds is not None:
            path = np.clip(path, self.space_bounds[0], self.space_bounds[1])

        repairs = 0
        validation_started = time.perf_counter()
        collisions = self.check_path_safety(path)
        validation_seconds = time.perf_counter() - validation_started
        hard_deadline = started + time_budget
        guide_seconds = 0.0
        while collisions and repairs < max_repairs:
            if time.perf_counter() >= hard_deadline:
                break
            segment_index = collisions[0]
            guide_started = time.perf_counter()
            bypass = self.realtime_guide.find_bypass(
                path[segment_index],
                path[segment_index + 1],
                point_collision=self.check_point_collision,
                segment_collision=self.check_segment_collision,
                bounds=self.space_bounds,
                distance_fractions=(0.35, 0.5, 0.65, 0.8),
                max_verifications=64,
                # Keep a small reserve for bookkeeping after exact 3-D
                # verification.  The guide also checks this deadline between
                # ray-cast obstacles and candidate segment checks.
                deadline=started + max(time_budget - 0.001, 0.0),
            )
            guide_seconds += time.perf_counter() - guide_started
            if bypass is None:
                break
            if time.perf_counter() >= hard_deadline:
                break
            path = np.insert(path, segment_index + 1, bypass, axis=0)
            repairs += 1
            # ``find_bypass`` has already verified both replacement segments.
            # All other segments are unchanged, so update their indices
            # instead of rescanning the complete path and obstacle set.
            collisions = [
                index + 1 if index > segment_index else index
                for index in collisions
                if index != segment_index
            ]

        elapsed = time.perf_counter() - started
        geometrically_safe = not collisions
        deadline_met = elapsed <= time_budget
        success = geometrically_safe and deadline_met
        self.last_realtime_stats = {
            "elapsed_seconds": elapsed,
            "deadline_seconds": float(time_budget),
            "deadline_met": deadline_met,
            "success": success,
            "geometrically_safe": geometrically_safe,
            "repairs": repairs,
            "remaining_collisions": len(collisions),
            "validation_seconds": validation_seconds,
            "guide_seconds": guide_seconds,
        }
        if success:
            self.path_points = path
            self.path_collisions = []
            return path
        return None

    def _fallback_replan(self, init_path, start, goal, corridors, max_attempts, smoothing):
        """启发式重规划: 逐步尝试将碰撞路径修正为安全路径。"""
        current_path = init_path.copy()
        current_collisions = self.check_path_safety(current_path)
        straight_length = np.linalg.norm(goal - start)

        if not current_collisions:
            return current_path

        for attempt in range(max_attempts):
            candidate = current_path.copy()

            if attempt == 0:
                # FIX: each waypoint only moves toward ITS OWN corridor center
                for j in range(1, len(candidate) - 1):
                    seg_idx = j - 1
                    if seg_idx not in current_collisions:
                        continue
                    if seg_idx >= len(corridors) or corridors[seg_idx] is None:
                        continue
                    _, ellipsoid = corridors[seg_idx]
                    dist = np.linalg.norm(candidate[j] - ellipsoid.center)
                    if dist > 1e-10:
                        direction = (ellipsoid.center - candidate[j]) / dist
                        # FIX: move TOWARD center, not past it
                        move_dist = min(dist * 0.5, 1.5)
                        candidate[j] = candidate[j] + direction * move_dist

            elif attempt == 1:
                best_candidate = candidate.copy()
                best_col_count = len(current_collisions)
                for _ in range(5):
                    temp = current_path.copy()
                    for idx in current_collisions:
                        if idx + 1 >= len(temp):
                            continue
                        path_dir = temp[idx + 1] - temp[idx]
                        perp_dir = np.cross(path_dir, np.random.randn(3))
                        norm = np.linalg.norm(perp_dir)
                        if norm > 1e-8:
                            perp_dir /= norm
                            wi = min(idx + 1, len(temp) - 2)
                            temp[wi] += perp_dir * np.linalg.norm(path_dir) * 0.3
                    temp_col = self.check_path_safety(temp)
                    if len(temp_col) < best_col_count:
                        best_candidate = temp.copy()
                        best_col_count = len(temp_col)
                candidate = best_candidate

            else:
                # FIX: track index offset when inserting points
                ins_offset = 0
                for idx in current_collisions:
                    real_idx = idx + ins_offset
                    if real_idx >= len(candidate) - 1:
                        continue
                    p1, p2 = candidate[real_idx], candidate[real_idx + 1]
                    mid = (p1 + p2) / 2
                    seg_dir = p2 - p1
                    ortho = np.array([1, 0, 0]) if abs(seg_dir[0]) < 0.9 else np.array([0, 1, 0])
                    perp = np.cross(seg_dir, ortho)
                    norm = np.linalg.norm(perp)
                    if norm > 1e-8:
                        perp /= norm
                        new_pt = mid + perp * 0.5
                        if not self.check_point_collision(new_pt):
                            candidate = np.insert(candidate, real_idx + 1, new_pt, axis=0)
                            ins_offset += 1

            new_col = self.check_path_safety(candidate)
            if not new_col:
                print("启发式重规划找到安全路径!")
                current_path = candidate
                current_collisions = []
                break
            elif len(new_col) < len(current_collisions):
                print(f"碰撞减少: {len(current_collisions)} -> {len(new_col)}")
                current_path = candidate
                current_collisions = new_col

        current_path = self._deduplicate_path(current_path)

        if smoothing and not self.check_path_safety(current_path):
            return current_path
        if smoothing:
            try:
                bs = self.bspline_smooth(current_path, smoothing_factor=0.9)
                if not self.check_path_safety(bs):
                    blen = np.sum(np.linalg.norm(np.diff(bs, axis=0), axis=1))
                    if blen > 1.3 * straight_length:
                        t = np.linspace(0, 1, len(bs))
                        sp = start * (1 - t[:, None]) + goal * t[:, None]
                        bs = 0.4 * bs + 0.6 * sp
                        if not self.check_path_safety(bs):
                            current_path = bs
                    else:
                        current_path = bs
            except Exception:
                pass
        return current_path
    
    def _collision_mask(self, points, safe_distance):
        """Vectorized collision decisions for one or many query points."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        collision = np.zeros(len(points), dtype=bool)

        if self._analytic_obstacles_complete:
            if len(self._sphere_centers):
                distance = np.linalg.norm(
                    points[:, None, :] - self._sphere_centers[None, :, :], axis=2
                ) - self._sphere_radii[None, :]
                collision |= np.any(distance < safe_distance, axis=1)

            if len(self._cylinder_centers):
                radial = np.linalg.norm(
                    points[:, None, :2]
                    - self._cylinder_centers[None, :, :2],
                    axis=2,
                ) - self._cylinder_radii[None, :]
                axial = np.abs(
                    points[:, None, 2]
                    - self._cylinder_centers[None, :, 2]
                ) - self._cylinder_heights[None, :] / 2.0
                outside = np.hypot(np.maximum(radial, 0.0), np.maximum(axial, 0.0))
                inside = np.minimum(np.maximum(radial, axial), 0.0)
                collision |= np.any(outside + inside < safe_distance, axis=1)

            if len(self._cuboid_centers):
                delta = (
                    np.abs(
                        points[:, None, :]
                        - self._cuboid_centers[None, :, :]
                    )
                    - self._cuboid_half_sizes[None, :, :]
                )
                outside = np.linalg.norm(np.maximum(delta, 0.0), axis=2)
                inside = np.minimum(np.max(delta, axis=2), 0.0)
                collision |= np.any(outside + inside < safe_distance, axis=1)

            if len(self._capsule_starts):
                axes = self._capsule_ends - self._capsule_starts
                denominator = np.einsum("ij,ij->i", axes, axes)
                relative = points[:, None, :] - self._capsule_starts[None, :, :]
                alpha = np.einsum("nci,ci->nc", relative, axes)
                alpha /= np.where(denominator > 1e-20, denominator, 1.0)[None, :]
                closest = (
                    self._capsule_starts[None, :, :]
                    + np.clip(alpha, 0.0, 1.0)[..., None] * axes[None, :, :]
                )
                distance = np.linalg.norm(points[:, None, :] - closest, axis=2)
                distance -= self._capsule_radii[None, :]
                collision |= np.any(distance < safe_distance, axis=1)
            return collision

        if self.obstacle_tree is not None and not self._analytic_obstacles_complete:
            distances, _ = self.obstacle_tree.query(points, k=1)
            collision |= distances < safe_distance

        for center, *params in self.obstacle_radii:
            center = np.asarray(center, dtype=float)
            if len(params) == 1:
                geometry = params[0]
                if np.isscalar(geometry) or (
                    isinstance(geometry, np.ndarray) and geometry.ndim == 0
                ):
                    signed = np.linalg.norm(points - center, axis=1) - float(
                        geometry
                    )
                else:
                    delta = np.abs(points - center) - np.asarray(geometry)
                    outside = np.linalg.norm(np.maximum(delta, 0.0), axis=1)
                    inside = np.minimum(np.max(delta, axis=1), 0.0)
                    signed = outside + inside
            elif len(params) == 2:
                radius, height = float(params[0]), float(params[1])
                radial = np.linalg.norm(points[:, :2] - center[:2], axis=1) - radius
                axial = np.abs(points[:, 2] - center[2]) - height / 2.0
                outside = np.linalg.norm(
                    np.maximum(np.column_stack((radial, axial)), 0.0), axis=1
                )
                inside = np.minimum(np.maximum(radial, axial), 0.0)
                signed = outside + inside
            else:
                continue
            collision |= signed < safe_distance
            if np.all(collision):
                break
        return collision

    def _sample_segment(self, p1, p2, samples=None):
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        if samples is None:
            samples = self.config.path_samples
        distance = np.linalg.norm(p2 - p1)
        if distance > 1.0:
            samples = max(samples, int(np.ceil(distance * 50)) + 1)
        t_values = np.linspace(0.0, 1.0, max(int(samples), 2))[:, None]
        return p1[None, :] * (1.0 - t_values) + p2[None, :] * t_values

    @staticmethod
    def _point_segment_distances_squared(points, start, end):
        """Squared Euclidean distances from points to a finite segment."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        start = np.asarray(start, dtype=float)
        end = np.asarray(end, dtype=float)
        direction = end - start
        denominator = float(np.dot(direction, direction))
        if denominator <= 1e-20:
            delta = points - start
            return np.einsum("ij,ij->i", delta, delta)
        alpha = ((points - start) @ direction) / denominator
        closest = start + np.clip(alpha, 0.0, 1.0)[:, None] * direction
        delta = points - closest
        return np.einsum("ij,ij->i", delta, delta)

    @staticmethod
    def _paired_segment_distances_squared(starts, ends, other_starts, other_ends):
        """Exact squared distances for equally sized pairs of 3-D segments."""
        starts = np.atleast_2d(np.asarray(starts, dtype=float))
        ends = np.atleast_2d(np.asarray(ends, dtype=float))
        other_starts = np.atleast_2d(np.asarray(other_starts, dtype=float))
        other_ends = np.atleast_2d(np.asarray(other_ends, dtype=float))
        if not (
            len(starts) == len(ends) == len(other_starts) == len(other_ends)
        ):
            raise ValueError("paired segment arrays must have the same length")
        if len(starts) == 0:
            return np.empty(0, dtype=float)

        query_axes = ends - starts
        other_axes = other_ends - other_starts
        relative = starts - other_starts
        query_norm_sq = np.einsum("ij,ij->i", query_axes, query_axes)
        other_norm_sq = np.einsum("ij,ij->i", other_axes, other_axes)
        cross_dot = np.einsum("ij,ij->i", other_axes, query_axes)
        query_relative_dot = np.einsum("ij,ij->i", relative, query_axes)
        other_relative_dot = np.einsum("ij,ij->i", other_axes, relative)
        best = np.full(len(other_starts), np.inf, dtype=float)

        def update(query_alpha, other_alpha):
            delta = (
                relative
                + np.asarray(query_alpha)[:, None] * query_axes
                - np.asarray(other_alpha)[:, None] * other_axes
            )
            distance_sq = np.einsum("ij,ij->i", delta, delta)
            np.minimum(best, distance_sq, out=best)

        safe_other_norm = np.where(other_norm_sq > 1e-20, other_norm_sq, 1.0)
        safe_query_norm = np.where(query_norm_sq > 1e-20, query_norm_sq, 1.0)
        count = len(other_starts)
        # Four boundary optima cover every constrained solution.  Add the
        # unconstrained interior optimum when it lies inside both segments.
        update(np.zeros(count), np.clip(other_relative_dot / safe_other_norm, 0.0, 1.0))
        update(
            np.ones(count),
            np.clip((other_relative_dot + cross_dot) / safe_other_norm, 0.0, 1.0),
        )
        update(
            np.clip(-query_relative_dot / safe_query_norm, 0.0, 1.0),
            np.zeros(count),
        )
        update(
            np.clip((cross_dot - query_relative_dot) / safe_query_norm, 0.0, 1.0),
            np.ones(count),
        )

        determinant = query_norm_sq * other_norm_sq - cross_dot * cross_dot
        nonparallel = determinant > 1e-20
        query_alpha = np.zeros(count)
        other_alpha = np.zeros(count)
        query_alpha[nonparallel] = (
            cross_dot[nonparallel] * other_relative_dot[nonparallel]
            - other_norm_sq[nonparallel] * query_relative_dot[nonparallel]
        ) / determinant[nonparallel]
        other_alpha[nonparallel] = (
            query_norm_sq * other_relative_dot[nonparallel]
            - cross_dot[nonparallel] * query_relative_dot[nonparallel]
        ) / determinant[nonparallel]
        interior = (
            nonparallel
            & (query_alpha >= 0.0)
            & (query_alpha <= 1.0)
            & (other_alpha >= 0.0)
            & (other_alpha <= 1.0)
        )
        if np.any(interior):
            delta = (
                relative[interior]
                + query_alpha[interior, None] * query_axes[interior]
                - other_alpha[interior, None] * other_axes[interior]
            )
            best[interior] = np.minimum(
                best[interior], np.einsum("ij,ij->i", delta, delta)
            )
        return best

    @staticmethod
    def _segment_segment_distances_squared(start, end, other_starts, other_ends):
        """Exact squared distances from one segment to many 3-D segments."""
        other_starts = np.atleast_2d(np.asarray(other_starts, dtype=float))
        count = len(other_starts)
        return FIRIPlanner._paired_segment_distances_squared(
            np.broadcast_to(np.asarray(start, dtype=float), (count, len(start))),
            np.broadcast_to(np.asarray(end, dtype=float), (count, len(end))),
            other_starts,
            other_ends,
        )

    def _analytic_segment_collision(self, p1, p2):
        """Resolve spheres, capsules and in-height trunk segments exactly.

        Returns ``(collision, fully_resolved)``.  Any cuboid, cap-adjacent
        cylinder segment or incomplete native obstacle set deliberately falls
        back to the sampled signed-distance implementation.
        """
        if not self._analytic_obstacles_complete:
            return False, False
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        fully_resolved = not len(self._cuboid_centers)

        if len(self._sphere_centers):
            distance_sq = self._point_segment_distances_squared(
                self._sphere_centers, p1, p2
            )
            thresholds = self._sphere_radii + self.safety_margin
            if np.any(distance_sq < thresholds * thresholds):
                return True, True

        if len(self._capsule_starts):
            distance_sq = self._segment_segment_distances_squared(
                p1, p2, self._capsule_starts, self._capsule_ends
            )
            thresholds = self._capsule_radii + self.safety_margin
            if np.any(distance_sq < thresholds * thresholds):
                return True, True

        if len(self._cylinder_centers):
            lower = self._cylinder_centers[:, 2] - self._cylinder_heights / 2.0
            upper = self._cylinder_centers[:, 2] + self._cylinder_heights / 2.0
            in_height = (
                (min(p1[2], p2[2]) >= lower)
                & (max(p1[2], p2[2]) <= upper)
            )
            if np.any(in_height):
                distance_sq = self._point_segment_distances_squared(
                    self._cylinder_centers[in_height, :2], p1[:2], p2[:2]
                )
                thresholds = (
                    self._cylinder_radii[in_height] + self.safety_margin
                )
                if np.any(distance_sq < thresholds * thresholds):
                    return True, True
            if not np.all(in_height):
                fully_resolved = False
        return False, fully_resolved

    def check_path_safety(self, path):
        """Return colliding segment indices using exact native geometry."""
        path = np.asarray(path, dtype=float)
        if len(path) < 2:
            return []
        starts = path[:-1]
        ends = path[1:]
        segment_count = len(starts)
        collision_flags = np.zeros(segment_count, dtype=bool)
        fully_resolved = np.full(
            segment_count,
            self._analytic_obstacles_complete and not len(self._cuboid_centers),
            dtype=bool,
        )

        if self._analytic_obstacles_complete:
            axes = ends - starts
            denominator = np.einsum("ij,ij->i", axes, axes)
            safe_denominator = np.where(denominator > 1e-20, denominator, 1.0)

            if len(self._sphere_centers):
                relative = self._sphere_centers[None, :, :] - starts[:, None, :]
                alpha = np.einsum("soi,si->so", relative, axes)
                alpha /= safe_denominator[:, None]
                closest = (
                    starts[:, None, :]
                    + np.clip(alpha, 0.0, 1.0)[..., None] * axes[:, None, :]
                )
                distance_sq = np.sum(
                    (self._sphere_centers[None, :, :] - closest) ** 2,
                    axis=2,
                )
                thresholds = self._sphere_radii + self.safety_margin
                collision_flags |= np.any(
                    distance_sq < thresholds[None, :] ** 2, axis=1
                )

            if len(self._capsule_starts):
                capsule_count = len(self._capsule_starts)
                distance_sq = self._paired_segment_distances_squared(
                    np.repeat(starts, capsule_count, axis=0),
                    np.repeat(ends, capsule_count, axis=0),
                    np.tile(self._capsule_starts, (segment_count, 1)),
                    np.tile(self._capsule_ends, (segment_count, 1)),
                ).reshape(segment_count, capsule_count)
                thresholds = self._capsule_radii + self.safety_margin
                collision_flags |= np.any(
                    distance_sq < thresholds[None, :] ** 2, axis=1
                )

            if len(self._cylinder_centers):
                lower = self._cylinder_centers[:, 2] - self._cylinder_heights / 2.0
                upper = self._cylinder_centers[:, 2] + self._cylinder_heights / 2.0
                in_height = (
                    np.minimum(starts[:, None, 2], ends[:, None, 2]) >= lower[None, :]
                ) & (
                    np.maximum(starts[:, None, 2], ends[:, None, 2]) <= upper[None, :]
                )
                relative_xy = (
                    self._cylinder_centers[None, :, :2] - starts[:, None, :2]
                )
                axes_xy = axes[:, :2]
                denominator_xy = np.einsum("ij,ij->i", axes_xy, axes_xy)
                alpha = np.einsum("soi,si->so", relative_xy, axes_xy)
                alpha /= np.where(
                    denominator_xy > 1e-20, denominator_xy, 1.0
                )[:, None]
                closest_xy = starts[:, None, :2] + np.clip(
                    alpha, 0.0, 1.0
                )[..., None] * axes_xy[:, None, :]
                radial_sq = np.sum(
                    (self._cylinder_centers[None, :, :2] - closest_xy) ** 2,
                    axis=2,
                )
                thresholds = self._cylinder_radii + self.safety_margin
                collision_flags |= np.any(
                    in_height & (radial_sq < thresholds[None, :] ** 2), axis=1
                )
                fully_resolved &= np.all(in_height, axis=1)

        collisions = np.flatnonzero(collision_flags).astype(int).tolist()
        sampled = []
        sampled_segment_ids = []
        for index in np.flatnonzero(~fully_resolved & ~collision_flags):
            points = self._sample_segment(starts[index], ends[index])
            sampled.append(points)
            sampled_segment_ids.append(np.full(len(points), index, dtype=int))
        if sampled:
            all_points = np.vstack(sampled)
            all_segment_ids = np.concatenate(sampled_segment_ids)
            collision_mask = self._collision_mask(all_points, self.safety_margin)
            if np.any(collision_mask):
                collisions.extend(
                    np.unique(all_segment_ids[collision_mask]).astype(int).tolist()
                )
        return sorted(set(collisions))
    
    def check_segment_collision(self, p1, p2, samples=None):
        analytic_collision, fully_resolved = self._analytic_segment_collision(
            p1, p2
        )
        if analytic_collision:
            return True
        if fully_resolved:
            return False
        points = self._sample_segment(p1, p2, samples=samples)
        return bool(np.any(self._collision_mask(points, self.safety_margin)))
    
    def check_point_collision(self, point, safe_distance=None):
        if safe_distance is None:
            safe_distance = self.safety_margin
        return bool(self._collision_mask(point, float(safe_distance))[0])
    
    def bspline_smooth(self, path, smoothing_factor=0.9):
        try:
            tck, u = splprep([path[:,0], path[:,1], path[:,2]], s=smoothing_factor)
            u_new = np.linspace(0, 1, len(path))
            smoothed = np.array(splev(u_new, tck)).T
            return smoothed
        except Exception as e:
            print(f"B样条平滑出错: {e}")
            return path

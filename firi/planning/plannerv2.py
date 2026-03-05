import numpy as np
import time
import os
import pickle
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from scipy.optimize import minimize
from .firi import FIRI
from .config import FIRIConfig
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
        
    def _build_obstacle_kdtree(self):
        vertices = []
        self.obstacle_radii = []
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
                elif shape == 'cylinder' and getattr(obs, 'height', None) is not None:
                    radius = obs.radius if obs.radius is not None else 1.0
                    height = obs.height
                    theta = np.linspace(0, 2 * np.pi, num_samples)
                    x = center[0] + radius * np.cos(theta)
                    y = center[1] + radius * np.sin(theta)
                    z = np.linspace(center[2] - height / 2, center[2] + height / 2, num_samples)
                    vertices.extend(np.column_stack((x, y, z)))
                    self.obstacle_radii.append((center, radius, height))
                else:
                    radius = obs.radius if obs.radius is not None else 1.0
                    theta = np.linspace(0, 2 * np.pi, num_samples)
                    x = center[0] + radius * np.cos(theta)
                    y = center[1] + radius * np.sin(theta)
                    z = np.linspace(center[2] - radius, center[2] + radius, num_samples)
                    vertices.extend(np.column_stack((x, y, z)))
                    self.obstacle_radii.append((center, radius))
            except Exception as e:
                print(f"构建KD树时处理障碍物出错: {e}")
                continue
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
        d_safe = self.config.d_safe
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
            _, ell = corr
            Q_inv = ell._cached_values.get('Q_inv', np.eye(dim))
            if np.any(np.isnan(Q_inv)) or np.any(np.isinf(Q_inv)):
                continue
            corridor_data.append((k, ell.center.copy(), Q_inv.copy()))

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

        for k, c_k, Q_inv_k in corridor_data:
            for t_s in [0.0, 0.5, 1.0]:
                def _corr(x, _k=k, _t=t_s, _c=c_k, _Qi=Q_inv_k):
                    pts = _all_pts(x)
                    if _k + 1 >= len(pts):
                        return 1.0
                    p = pts[_k] * (1 - _t) + pts[_k + 1] * _t
                    diff = p - _c
                    return 1.0 - float(diff @ _Qi @ diff)
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
            optimized = _all_pts(result.x)
            if result.success:
                print(f"轨迹优化收敛 (iter={result.nit}, cost={result.fun:.4f})")
            else:
                print(f"轨迹优化未完全收敛: {result.message}")

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
    # 主规划流程 (按伪代码 Algorithm 1 重构)
    # =====================================================================
    def plan_path(self, start, goal, initial_waypoints=None, smoothing=True,
                  max_replanning_attempts=10, safety_margin=1.0):
        os.makedirs('temp', exist_ok=True)
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
                init_path, start, goal, corridors, max_replanning_attempts, smoothing)
            collisions = self.check_path_safety(final_path)

        # 裁剪到边界
        if self.space_bounds is not None:
            final_path = np.clip(
                final_path, self.space_bounds[0], self.space_bounds[1])

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
            pickle.dump(final_path, open('temp/path_points.pkl', 'wb'))
            pickle.dump(final_path, open('temp/adjusted_path.pkl', 'wb'))
        except Exception:
            pass
        return final_path

    def _fallback_replan(self, init_path, start, goal, corridors, max_attempts, smoothing):
        """旧版启发式重规划 (作为安全回退)。"""
        collisions = self.check_path_safety(init_path)
        straight_length = np.linalg.norm(goal - start)

        for attempt in range(max_attempts):
            replan_path = init_path.copy()
            if attempt == 0:
                for i, region in enumerate(corridors):
                    if region is None:
                        continue
                    _, ellipsoid = region
                    for j in range(1, len(replan_path) - 1):
                        if j - 1 in collisions:
                            dist = np.linalg.norm(replan_path[j] - ellipsoid.center)
                            move_dist = min(dist * 0.8, 1.5)
                            if dist > 1e-10:
                                direction = (ellipsoid.center - replan_path[j]) / dist
                                replan_path[j] = ellipsoid.center + direction * move_dist
            elif attempt == 1:
                best_path = replan_path.copy()
                best_collisions = collisions
                for _ in range(5):
                    temp_path = init_path.copy()
                    for idx in collisions:
                        path_dir = temp_path[min(idx + 1, len(temp_path) - 1)] - temp_path[idx]
                        perp_dir = np.cross(path_dir, np.random.randn(3))
                        norm = np.linalg.norm(perp_dir)
                        if norm > 1e-8:
                            perp_dir /= norm
                            temp_path[idx] += perp_dir * np.linalg.norm(path_dir) * 0.3
                    temp_col = self.check_path_safety(temp_path)
                    if len(temp_col) < len(best_collisions):
                        best_path, best_collisions = temp_path, temp_col
                replan_path = best_path
                collisions = best_collisions
            else:
                for idx in collisions:
                    if idx >= len(replan_path) - 1:
                        continue
                    mid = (replan_path[idx] + replan_path[idx + 1]) / 2
                    offset = replan_path[idx + 1] - replan_path[idx]
                    perp = np.cross(offset, np.array([0, 0, 1]))
                    norm = np.linalg.norm(perp)
                    if norm > 1e-8:
                        perp /= norm
                        new_pt = mid + perp * 0.5
                        replan_path = np.insert(replan_path, idx + 1, new_pt, axis=0)

            new_col = self.check_path_safety(replan_path)
            if not new_col:
                print("启发式重规划找到安全路径!")
                init_path = replan_path
                break
            elif len(new_col) < len(collisions):
                print(f"碰撞减少: {len(collisions)} -> {len(new_col)}")
                init_path = replan_path
                collisions = new_col

        final = init_path
        if smoothing and not self.check_path_safety(final):
            return final
        if smoothing:
            try:
                bs = self.bspline_smooth(final, smoothing_factor=0.9)
                if not self.check_path_safety(bs):
                    blen = np.sum(np.linalg.norm(np.diff(bs, axis=0), axis=1))
                    if blen > 1.3 * straight_length:
                        t = np.linspace(0, 1, len(bs))
                        sp = start * (1 - t[:, None]) + goal * t[:, None]
                        bs = 0.4 * bs + 0.6 * sp
                        if not self.check_path_safety(bs):
                            final = bs
                    else:
                        final = bs
            except Exception:
                pass
        return final
    
    def check_path_safety(self, path):
        collisions = []
        for i in range(len(path) - 1):
            if self.check_segment_collision(path[i], path[i+1]):
                collisions.append(i)
        return collisions
    
    def check_segment_collision(self, p1, p2, samples=None):
        if samples is None:
            samples = self.config.path_samples
        dist = np.linalg.norm(p2 - p1)
        if dist > 1.0:
            samples = max(samples, int(dist * 50))
        t_values = np.linspace(0, 1, samples)
        for t in t_values:
            point = p1 * (1-t) + p2 * t
            if self.check_point_collision(point):
                return True
        return False
    
    def check_point_collision(self, point, safe_distance=None):
        if safe_distance is None:
            safe_distance = 0.3
        if self.obstacle_tree is not None:
            distance, _ = self.obstacle_tree.query(point, k=1)
            if distance < safe_distance:
                return True
        for center, *params in self.obstacle_radii:
            dist = float('inf')
            if len(params) == 1:
                p = params[0]
                if np.isscalar(p) or (isinstance(p, np.ndarray) and p.ndim == 0):
                    dist = np.linalg.norm(point - center) - float(p)
                else:
                    dist = float(np.max(np.abs(point - center) - np.array(p)))
            elif len(params) == 2:
                radius, height = float(params[0]), float(params[1])
                xy_dist = np.linalg.norm(point[:2] - center[:2]) - radius
                z_dist = abs(point[2] - center[2]) - height / 2
                dist = max(xy_dist, z_dist)
            if dist < safe_distance:
                return True
        return False
    
    def bspline_smooth(self, path, smoothing_factor=0.9):
        try:
            tck, u = splprep([path[:,0], path[:,1], path[:,2]], s=smoothing_factor)
            u_new = np.linspace(0, 1, len(path))
            smoothed = np.array(splev(u_new, tck)).T
            return smoothed
        except Exception as e:
            print(f"B样条平滑出错: {e}")
            return path
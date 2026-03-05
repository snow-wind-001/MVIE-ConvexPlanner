import numpy as np
import time
import os
import pickle
from scipy.spatial import KDTree
from scipy.interpolate import splprep, splev
from .firi import FIRI
from .config import FIRIConfig
from ..geometry import Ellipsoid

class FIRIPlanner:
    def __init__(self, obstacles, space_size):
        self.obstacles = obstacles
        self.space_size = space_size
        self.dimension = len(space_size)
        self.config = FIRIConfig(space_size)
        self.config.update_adaptive_params(obstacle_count=len(obstacles))
        self.firi = FIRI(obstacles, self.dimension)
        self._build_obstacle_kdtree()
        self.safe_regions = []
        self.path_points = []
        self.path_collisions = []
        
    def _build_obstacle_kdtree(self):
        vertices = []
        self.obstacle_radii = []
        for obs in self.obstacles:
            try:
                if hasattr(obs, 'center') and hasattr(obs, 'radius'):
                    center = np.array(obs.center)
                    radius = obs.radius if obs.radius is not None else 1.0
                    num_samples = 50
                    theta = np.linspace(0, 2 * np.pi, num_samples)
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    z = np.linspace(center[2] - radius, center[2] + radius, num_samples)
                    vertices.extend(np.column_stack((x, y, z)))
                    self.obstacle_radii.append((center, radius))
                elif hasattr(obs, 'size'):
                    center = np.array(obs.center)
                    size = np.array(obs.size)
                    half_sizes = size / 2
                    x = np.array([-half_sizes[0], half_sizes[0], half_sizes[0], -half_sizes[0], -half_sizes[0], half_sizes[0], half_sizes[0], -half_sizes[0]])
                    y = np.array([-half_sizes[1], -half_sizes[1], half_sizes[1], half_sizes[1], -half_sizes[1], -half_sizes[1], half_sizes[1], half_sizes[1]])
                    z = np.array([-half_sizes[2], -half_sizes[2], -half_sizes[2], -half_sizes[2], half_sizes[2], half_sizes[2], half_sizes[2], half_sizes[2]])
                    vertices.extend(np.column_stack((x, y, z)) + center)
                    self.obstacle_radii.append((center, half_sizes))
                elif hasattr(obs, 'height'):
                    center = np.array(obs.center)
                    radius = obs.radius if obs.radius is not None else 1.0
                    height = obs.height if obs.height is not None else 1.0
                    self.obstacle_radii.append((center, radius, height))
                    num_samples = 50
                    theta = np.linspace(0, 2 * np.pi, num_samples)
                    x = radius * np.cos(theta)
                    y = radius * np.sin(theta)
                    z = np.linspace(center[2] - height / 2, center[2] + height / 2, num_samples)
                    vertices.extend(np.column_stack((x, y, z)))
            except Exception as e:
                print(f"构建KD树时处理障碍物出错: {e}")
                continue
        if vertices:
            self.obstacle_tree = KDTree(vertices)
            print(f"已构建KD-Tree: {len(vertices)}个顶点")
        else:
            self.obstacle_tree = None
            print("警告: 无法构建障碍物KD树")
    
    def generate_safe_regions(self, start, goal, num_waypoints=4):
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
    
    def generate_initial_waypoints(self, start, goal, num_waypoints=4, jitter_ratio=0.08):
        """
        生成更合理的初始路径点，扰动仅在主方向的垂直平面内，且扰动幅度在路径中间最大，两端为0
        jitter_ratio: 扰动占总路径长度的比例（如0.08表示8%）
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

    def plan_path(self, start, goal, initial_waypoints=None, smoothing=True, max_replanning_attempts=7, safety_margin=1.0):
        if initial_waypoints is None:
            num_waypoints = 4
            # 使用改进的初始路径生成方式
            init_path = self.generate_initial_waypoints(start, goal, num_waypoints=num_waypoints, jitter_ratio=0.08)
        else:
            init_path = np.array(initial_waypoints)
        collisions = self.check_path_safety(init_path)
        if not collisions:
            print("初始路径安全，无需重新规划")
            self.path_points = init_path
            if smoothing:
                try:
                    bspline_path = self.bspline_smooth(init_path, smoothing_factor=0.3)
                    bspline_collisions = self.check_path_safety(bspline_path)
                    if not bspline_collisions:
                        print("路径平滑成功且安全")
                        init_path = bspline_path
                    else:
                        print(f"平滑路径不安全，有 {len(bspline_collisions)} 处碰撞，使用原始路径")
                except Exception as e:
                    print(f"B样条平滑出错: {e}")
            self.path_points = init_path
            try:
                pickle.dump(init_path, open('temp/path_points.pkl', 'wb'))
                with open('temp/path_safety.txt', 'w') as f:
                    f.write(f"path_points: {len(init_path)}\n")
                    f.write(f"collision_segments: {len(collisions)}\n")
                    f.write(f"collision_indices: {collisions}\n")
                    f.write(f"path_safety: {'Safe' if not collisions else 'Unsafe'}\n")
            except Exception as e:
                print(f"保存路径信息出错: {e}")
            return init_path
        print(f"发现碰撞! 尝试重新规划路径... (初始碰撞段: {collisions})")
        self.generate_safe_regions(start, goal)
        for attempt in range(max_replanning_attempts):
            replan_path = init_path.copy()
            used_zigzag = False
            if attempt == 0:
                for i, region in enumerate(self.safe_regions):
                    if region is None:
                        continue
                    _, ellipsoid = region
                    for j in range(1, len(replan_path)-1):
                        if j-1 in collisions:
                            dist = np.linalg.norm(replan_path[j] - ellipsoid.center)
                            move_dist = min(dist * 0.8, 4.0)
                            if dist > 1e-10:
                                direction = (ellipsoid.center - replan_path[j]) / dist
                                replan_path[j] += direction * move_dist
            elif attempt == 1:
                best_path = replan_path.copy()
                best_collisions = collisions
                for sub_attempt in range(3):
                    temp_path = init_path.copy()
                    for idx in collisions:
                        path_dir = temp_path[idx+1] - temp_path[idx]
                        path_length = np.linalg.norm(path_dir)
                        perp_dir = np.cross(path_dir, np.random.randn(3))
                        perp_dir /= np.linalg.norm(perp_dir)
                        displacement = perp_dir * path_length * 0.5
                        temp_path[idx] += displacement
                    temp_collisions = self.check_path_safety(temp_path)
                    if len(temp_collisions) < len(best_collisions):
                        best_path = temp_path
                        best_collisions = temp_collisions
                replan_path = best_path
                collisions = best_collisions
            else:
                if len(collisions) > 0:
                    if len(replan_path) >= 3:
                        midpoint = (start + goal) / 2
                        offset_amount = np.linalg.norm(goal - start) * 0.4
                        path_dir = goal - start
                        path_dir = path_dir / np.linalg.norm(path_dir)
                        if np.abs(path_dir[0]) < np.abs(path_dir[1]):
                            perp_dir = np.array([1, 0, 0])
                        else:
                            perp_dir = np.array([0, 1, 0])
                        perp_dir = perp_dir - np.dot(perp_dir, path_dir) * path_dir
                        perp_dir = perp_dir / np.linalg.norm(perp_dir)
                        if len(replan_path) >= 3:
                            zigzag_path = np.zeros_like(replan_path)
                            zigzag_path[0] = start
                            zigzag_path[-1] = goal
                            for i in range(1, len(zigzag_path)-1):
                                t = i / (len(zigzag_path)-1)
                                zigzag_path[i] = start * (1-t) + goal * t
                                if i % 2 == 1:
                                    zigzag_path[i] += perp_dir * offset_amount
                                else:
                                    zigzag_path[i] -= perp_dir * offset_amount
                            replan_path = zigzag_path
                            used_zigzag = True
            new_collisions = self.check_path_safety(replan_path)
            if not new_collisions:
                print("找到安全路径!")
                init_path = replan_path
                if used_zigzag:
                    print("使用B样条拟合Z字形路径...")
                    bspline_path = self.bspline_smooth(init_path, smoothing_factor=0.3)
                    bspline_collisions = self.check_path_safety(bspline_path)
                    if not bspline_collisions:
                        print("B样条路径安全，采用拟合后的轨迹")
                        init_path = bspline_path
                    else:
                        print(f"B样条轨迹不安全，继续使用Z字形路径（碰撞段: {bspline_collisions}）")
                break
            elif len(new_collisions) < len(collisions):
                print(f"碰撞减少: {len(collisions)} -> {len(new_collisions)}")
                init_path = replan_path
                collisions = new_collisions
        self.path_collisions = collisions
        if collisions:
            print(f"警告: 路径规划未能完全消除碰撞，仍有 {len(collisions)} 处碰撞")
        final_path = init_path
        if smoothing and not collisions and not used_zigzag:
            try:
                bspline_path = self.bspline_smooth(final_path, smoothing_factor=0.3)
                bspline_collisions = self.check_path_safety(bspline_path)
                if not bspline_collisions:
                    print("路径平滑成功且安全")
                    final_path = bspline_path
                else:
                    print(f"平滑路径不安全，有 {len(bspline_collisions)} 处碰撞，使用原始路径")
            except Exception as e:
                print(f"B样条平滑出错: {e}")
        self.path_points = final_path
        print("最终路径点:", final_path)
        try:
            pickle.dump(final_path, open('temp/path_points.pkl', 'wb'))
            pickle.dump(final_path, open('temp/adjusted_path.pkl', 'wb'))
            with open('temp/path_safety.txt', 'w') as f:
                f.write(f"path_points: {len(final_path)}\n")
                f.write(f"collision_segments: {len(collisions)}\n")
                f.write(f"collision_indices: {collisions}\n")
                f.write(f"path_safety: {'Safe' if not collisions else 'Unsafe'}\n")
        except Exception as e:
            print(f"保存路径信息出错: {e}")
        return final_path
    
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
            samples = max(samples, int(dist * 10))
        t_values = np.linspace(0, 1, samples)
        for t in t_values:
            point = p1 * (1-t) + p2 * t
            if self.check_point_collision(point):
                return True
        return False
    
    def check_point_collision(self, point, safe_distance=None):
        if safe_distance is None:
            safe_distance = self.config.collision_threshold
        if self.obstacle_tree is not None:
            distance, _ = self.obstacle_tree.query(point, k=1)
            if distance < safe_distance:
                return True
        for center, *params in self.obstacle_radii:
            if len(params) == 1:
                radius = params[0]
                dist = np.linalg.norm(point - center) - radius
            elif len(params) == 2:
                half_sizes = params[0]
                dist = np.abs(point - center) - half_sizes
                dist = np.max(dist)
            elif len(params) == 3:
                radius, height = params[0], params[1]
                dist = np.linalg.norm(point - center[:2]) - radius
                if dist < 0 and center[2] - height / 2 <= point[2] <= center[2] + height / 2:
                    return True
            if dist < safe_distance:
                return True
        return False
    
    def bspline_smooth(self, path, smoothing_factor=0.3):
        try:
            tck, u = splprep([path[:,0], path[:,1], path[:,2]], s=smoothing_factor)
            u_new = np.linspace(0, 1, len(path))
            smoothed = np.array(splev(u_new, tck)).T
            return smoothed
        except Exception as e:
            print(f"B样条平滑出错: {e}")
            return path
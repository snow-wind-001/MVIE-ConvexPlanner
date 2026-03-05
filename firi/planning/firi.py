import numpy as np
from ..geometry import Ellipsoid, ConvexPolytope
from .mvie import MVIE_SOCP

class FIRI:
    def __init__(self, obstacles, dimension=3, space_bounds=None):
        self.obstacles = obstacles
        self.dim = dimension
        self.mvie_socp = MVIE_SOCP(dimension)
        self.space_bounds = space_bounds
    
    def compute_safe_region(self, seed_points, initial_ellipsoid=None, 
                           max_iterations=5, volume_threshold=0.05):
        seed_points = np.array(seed_points)
        if len(seed_points) == 0:
            raise ValueError("需要至少一个种子点")
        
        center = np.mean(seed_points, axis=0)
        seed_extent = np.linalg.norm(np.max(seed_points, axis=0) - np.min(seed_points, axis=0))
        Q = np.eye(self.dim) * max(0.5, seed_extent**2 / 20)
        current_ellipsoid = Ellipsoid(center, Q)
        
        prev_volume = current_ellipsoid.volume()
        last_valid_ellipsoid = current_ellipsoid
        last_valid_polytope = None
        abnormal_growth_count = 0
        
        for iter_num in range(max_iterations):
            print(f"FIRI迭代 {iter_num+1}/{max_iterations}...")
            try:
                current_polytope = self.restrictive_inflation(current_ellipsoid, seed_points)
                new_ellipsoid = self.mvie_socp.compute(current_polytope)
                current_volume = new_ellipsoid.volume()
                
                volume_increase = (current_volume - prev_volume) / max(prev_volume, 1e-10)
                print(f"  当前椭球体体积: {current_volume:.6f}, 增长比例: {volume_increase:.2%}")
                
                if volume_increase > 5.0 or current_volume > 1000:
                    abnormal_growth_count += 1
                    print(f"  警告: 检测到异常体积增长 ({abnormal_growth_count}/2)")
                    if abnormal_growth_count > 1:
                        print("  体积异常，回退到上次有效椭球体")
                        return last_valid_polytope, last_valid_ellipsoid
                else:
                    abnormal_growth_count = 0
                
                if abs(volume_increase) < volume_threshold:
                    print("  已收敛，停止迭代")
                    return current_polytope, new_ellipsoid
                
                current_ellipsoid = new_ellipsoid
                last_valid_ellipsoid = new_ellipsoid
                last_valid_polytope = current_polytope
                prev_volume = current_volume
            except Exception as e:
                print(f"  FIRI迭代 {iter_num+1} 失败: {e}")
                break
        
        return last_valid_polytope, last_valid_ellipsoid
    
    def restrictive_inflation(self, ellipsoid, seed_points):
        standard_halfspaces = []
        valid_obstacles = []
        
        for obs in self.obstacles:
            try:
                if hasattr(obs, 'center'):
                    obs_center = np.array(obs.center).ravel()
                elif isinstance(obs, dict) and 'center' in obs:
                    obs_center = np.array(obs['center']).ravel()
                else:
                    continue

                shape = getattr(obs, 'shape', None)
                if shape == 'cuboid' and getattr(obs, 'size', None) is not None:
                    obs_radius = float(np.linalg.norm(np.array(obs.size)) / 2)
                elif shape == 'cylinder' and getattr(obs, 'height', None) is not None:
                    obs_radius = float(max(obs.radius, obs.height / 2))
                elif hasattr(obs, 'radius') and obs.radius is not None:
                    obs_radius = float(obs.radius)
                elif isinstance(obs, dict) and 'radius' in obs:
                    obs_radius = float(obs['radius'])
                else:
                    obs_radius = 1.0
            except Exception as e:
                print(f"  障碍物处理出错: {e}")
                continue

            min_dist_to_seed = float('inf')
            for seed in seed_points:
                seed_arr = np.array(seed).ravel()
                dist = np.linalg.norm(seed_arr - obs_center) - obs_radius
                min_dist_to_seed = min(min_dist_to_seed, dist)

            if min_dist_to_seed < 3.0 * obs_radius:
                valid_obstacles.append((obs_center, obs_radius, min_dist_to_seed))
    
        valid_obstacles.sort(key=lambda x: x[2])
        max_obstacles = min(30, len(valid_obstacles))
        valid_obstacles = valid_obstacles[:max_obstacles]
        print(f"  找到 {len(valid_obstacles)} 个有效障碍物")
        
        seed_arr_list = [np.array(s).ravel() for s in seed_points]
        seed_center = np.mean(seed_arr_list, axis=0)

        for obs_center, obs_radius, _ in valid_obstacles:
            margin = max(0.05, obs_radius * 0.1)

            best_hs = None
            best_count = -1
            for ref_point in seed_arr_list + [seed_center]:
                diff = ref_point - obs_center
                dist = np.linalg.norm(diff)
                if dist < 1e-10:
                    diff = np.random.randn(self.dim)
                    dist = np.linalg.norm(diff)
                n = diff / dist
                hs = np.zeros(self.dim + 1)
                hs[:-1] = -n
                hs[-1] = np.dot(n, obs_center) + obs_radius + margin
                ok_count = sum(1 for s in seed_arr_list if float(np.dot(hs[:-1], s) + hs[-1]) <= 1e-6)
                if ok_count > best_count:
                    best_count = ok_count
                    best_hs = hs

            if best_hs is not None and best_count > 0:
                standard_halfspaces.append(best_hs)
    
        if self.space_bounds is not None:
            bbox_min = np.array(self.space_bounds[0], dtype=float)
            bbox_max = np.array(self.space_bounds[1], dtype=float)
        else:
            seed_min = np.min(np.array(seed_points), axis=0)
            seed_max = np.max(np.array(seed_points), axis=0)
            seed_extent = max(np.linalg.norm(seed_max - seed_min), 1.0)
            bbox_min = seed_min - seed_extent
            bbox_max = seed_max + seed_extent
        for i in range(self.dim):
            min_hs = np.zeros(self.dim + 1)
            min_hs[i] = -1
            min_hs[-1] = bbox_min[i]
            standard_halfspaces.append(min_hs)
            max_hs = np.zeros(self.dim + 1)
            max_hs[i] = 1
            max_hs[-1] = -bbox_max[i]
            standard_halfspaces.append(max_hs)
    
        original_halfspaces = list(standard_halfspaces)
        
        for seed in seed_arr_list:
            violating = []
            for idx, hs in enumerate(original_halfspaces):
                val = float(np.dot(hs[:-1], seed) + hs[-1])
                if val > 1e-6:
                    violating.append((idx, val))
            for idx, val in violating:
                original_halfspaces[idx][-1] -= (val + 0.1)

        original_polytope = ConvexPolytope(halfspaces=np.array(original_halfspaces))
        seed_contained_count = sum(1 for s in seed_arr_list if original_polytope.contains(s))
        print(f"  多胞体包含 {seed_contained_count}/{len(seed_points)} 个种子点")
        
        return original_polytope
    
    def compute_mvie(self, polytope):
        return self.mvie_socp.compute(polytope)
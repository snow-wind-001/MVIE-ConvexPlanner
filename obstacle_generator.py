import numpy as np
import pickle
import os

class Obstacle:
    def __init__(self, shape, center, radius=None, size=None, height=None):
        self.shape = shape  # 'sphere', 'cylinder', 'cuboid'
        self.center = center
        self.radius = radius if radius is not None else 1.0  # 如果未指定半径，则默认为1.0
        self.size = size  # For cuboid: (length, width, height)
        self.height = height  # For cylinder: height

class ObstacleSet:
    def __init__(self):
        self.obstacle_list = []

    def add_obstacle(self, shape, center, radius=None, size=None, height=None):
        self.obstacle_list.append(Obstacle(shape, center, radius, size, height))

    def __len__(self):
        return len(self.obstacle_list)

    def __iter__(self):
        return iter(self.obstacle_list)

    def check_collision(self, new_obs):
        """检查新障碍物是否与已有障碍物碰撞"""
        for obs in self.obstacle_list:
            if obs.shape == 'sphere' and new_obs.shape == 'sphere':
                dist = np.linalg.norm(np.array(obs.center) - np.array(new_obs.center))
                if dist < obs.radius + new_obs.radius:
                    return True
            elif obs.shape == 'cylinder' and new_obs.shape == 'cylinder':
                dist = np.linalg.norm(np.array(obs.center) - np.array(new_obs.center))
                if dist < obs.radius + new_obs.radius:
                    return True
            elif obs.shape == 'cuboid' and new_obs.shape == 'cuboid':
                dist = np.linalg.norm(np.array(obs.center) - np.array(new_obs.center))
                if dist < np.linalg.norm(obs.size / 2 + new_obs.size / 2):
                    return True
        return False

def place_obstacles(space_boundary, start, goal,
                    n_spheres=3, n_cylinders=2, n_cuboids=3,
                    density='medium', num_on_path=2,
                    safe_radius_start=1.5, safe_radius_goal=1.5):
    """
    在仿真空间中按类型生成指定数量的障碍物。

    参数:
        space_boundary: [[x_min,x_max],[y_min,y_max],[z_min,z_max]]
        start, goal:    起点/终点坐标
        n_spheres:      球体数量
        n_cylinders:    圆柱体数量
        n_cuboids:      长方体数量
        density:        障碍物密度 ('low'|'medium'|'high') — 影响尺寸和间距
        num_on_path:    放置在路径连线上的球体障碍物数量（从 n_spheres 中扣除）
        safe_radius_start: 起点周围的安全半径（不放障碍物）
        safe_radius_goal:  终点周围的安全半径
    """
    obstacles = ObstacleSet()

    size_table = {
        'low':    {'radius': (0.3, 0.7),  'height': (0.5, 1.2), 'cuboid': (0.3, 0.8)},
        'medium': {'radius': (0.5, 1.2),  'height': (0.8, 1.8), 'cuboid': (0.5, 1.3)},
        'high':   {'radius': (0.8, 1.5),  'height': (1.0, 2.5), 'cuboid': (0.7, 1.8)},
    }
    sz = size_table.get(density, size_table['medium'])

    direction = goal - start
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        direction_unit = direction / direction_norm
    else:
        direction_unit = np.array([0, 1, 0])

    def _too_close_to_endpoints(pos, radius_eff):
        if np.linalg.norm(pos - start) < safe_radius_start + radius_eff:
            return True
        if np.linalg.norm(pos - goal) < safe_radius_goal + radius_eff:
            return True
        return False

    actual_on_path = min(num_on_path, n_spheres)
    for _ in range(actual_on_path):
        t = np.random.uniform(0.25, 0.75)
        pos = start + t * (goal - start)
        r = np.random.uniform(*sz['radius'])
        if not _too_close_to_endpoints(pos, r):
            obstacles.add_obstacle('sphere', pos, radius=r)
            print(f"[路径上] 球体 center={np.round(pos,2)}, r={r:.2f}")

    remaining_spheres = n_spheres - actual_on_path

    build_list = (
        [('sphere', remaining_spheres)] +
        [('cylinder', n_cylinders)] +
        [('cuboid', n_cuboids)]
    )

    max_attempts = 200
    for shape, count in build_list:
        for _ in range(count):
            for attempt in range(max_attempts):
                x = np.random.uniform(space_boundary[0][0], space_boundary[0][1])
                y = np.random.uniform(space_boundary[1][0], space_boundary[1][1])
                z = np.random.uniform(space_boundary[2][0], space_boundary[2][1])
                pos = np.array([x, y, z])

                if direction_norm > 1e-6:
                    dist_to_path = np.linalg.norm(np.cross(direction_unit, pos - start))
                    if dist_to_path > 4.0:
                        continue

                if shape == 'sphere':
                    r = np.random.uniform(*sz['radius'])
                    new_obs = Obstacle('sphere', pos, radius=r)
                    eff_r = r
                elif shape == 'cylinder':
                    r = np.random.uniform(*sz['radius'])
                    h = np.random.uniform(*sz['height'])
                    new_obs = Obstacle('cylinder', pos, radius=r, height=h)
                    eff_r = max(r, h / 2)
                else:
                    s = np.random.uniform(*sz['cuboid'], size=3)
                    new_obs = Obstacle('cuboid', pos, size=s)
                    eff_r = np.linalg.norm(s) / 2

                if _too_close_to_endpoints(pos, eff_r):
                    continue
                if not obstacles.check_collision(new_obs):
                    obstacles.add_obstacle(
                        new_obs.shape, new_obs.center,
                        radius=new_obs.radius,
                        height=getattr(new_obs, 'height', None),
                        size=getattr(new_obs, 'size', None)
                    )
                    print(f"[{shape}] center={np.round(pos,2)}, "
                          f"r={new_obs.radius:.2f}, h={getattr(new_obs,'height',None)}, "
                          f"size={np.round(new_obs.size,2) if new_obs.size is not None else None}")
                    break

    total = len(obstacles.obstacle_list)
    type_counts = {}
    for o in obstacles.obstacle_list:
        type_counts[o.shape] = type_counts.get(o.shape, 0) + 1
    print(f"障碍物生成完成: 共{total}个 — {type_counts}")
    return obstacles

def save_obstacles_to_file(obstacles):
    """保存障碍物信息到文件"""
    with open('temp/obstacles.pkl', 'wb') as f:
        pickle.dump(obstacles, f)
    print(f"Saved {len(obstacles.obstacle_list)} obstacles to file")
    

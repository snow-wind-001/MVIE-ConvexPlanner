import os
import pickle
import numpy as np
import time
import json
from datetime import datetime
from performance_evaluator import PerformanceEvaluator
from obstacle_generator import ObstacleSet, place_obstacles, save_obstacles_to_file
from path_planner import generate_initial_waypoints, calculate_path_length
from visualizer import visualize_results, visualize_with_open3d, visualize_interactive
from utils import analyze_path_smoothness, check_collisions, analyze_path_results
from firi.planning.plannerv2 import FIRIPlanner
from firi.utils.obstacle_generator import ObstacleGenerator

def clean_temp_dir():
    """清理临时目录"""
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    print("Temp directory cleaned")

def generate_random_endpoints(space_bounds, margin=1.0):
    """
    在仿真空间最长轴的两端随机生成起点和终点。
    起点在低端区域，终点在高端区域，其余轴坐标随机。
    margin: 距离边界的最小安全裕度
    """
    bounds_min = space_bounds[0]
    bounds_max = space_bounds[1]
    extents = bounds_max - bounds_min
    long_axis = int(np.argmax(extents))

    start_point = np.zeros(3)
    goal_point = np.zeros(3)
    for i in range(3):
        lo = bounds_min[i] + margin
        hi = bounds_max[i] - margin
        if lo >= hi:
            lo, hi = bounds_min[i], bounds_max[i]
        if i == long_axis:
            start_point[i] = np.random.uniform(lo, lo + extents[i] * 0.1)
            goal_point[i] = np.random.uniform(hi - extents[i] * 0.1, hi)
        else:
            start_point[i] = np.random.uniform(lo, hi)
            goal_point[i] = np.random.uniform(lo, hi)
    return start_point, goal_point


def _find_bypass_point(p1, p2, planner, bounds_min, bounds_max):
    """为碰撞段找到安全绕行点。尝试多个锚点位置和更大搜索范围。"""
    seg_dir = p2 - p1
    seg_len = np.linalg.norm(seg_dir)
    if seg_len < 1e-8:
        return None

    seg_unit = seg_dir / seg_len
    ortho = np.array([1, 0, 0]) if abs(seg_unit[0]) < 0.9 else np.array([0, 1, 0])
    perp1 = np.cross(seg_unit, ortho)
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(seg_unit, perp1)

    best_candidate = None
    best_cost = np.inf

    for anchor_t in [0.5, 0.35, 0.65, 0.2, 0.8]:
        mid = p1 * (1 - anchor_t) + p2 * anchor_t
        for angle in np.linspace(0, 2 * np.pi, 16, endpoint=False):
            for dist_mult in [0.5, 0.8, 1.2, 1.8, 2.5, 3.5, 4.5]:
                disp = dist_mult * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
                candidate = mid + disp
                candidate = np.clip(candidate, bounds_min + 0.15, bounds_max - 0.15)

                if planner.check_point_collision(candidate):
                    continue

                if (not planner.check_segment_collision(p1, candidate) and
                        not planner.check_segment_collision(candidate, p2)):
                    cost = np.linalg.norm(p1 - candidate) + np.linalg.norm(candidate - p2)
                    if cost < best_cost:
                        best_candidate = candidate.copy()
                        best_cost = cost

        if best_candidate is not None:
            return best_candidate

    return best_candidate


def _find_two_point_bypass(p1, p2, planner, bounds_min, bounds_max):
    """当单点绕行失败时，尝试插入两个绕行点。"""
    seg_dir = p2 - p1
    seg_len = np.linalg.norm(seg_dir)
    if seg_len < 1e-8:
        return None

    seg_unit = seg_dir / seg_len
    ortho = np.array([1, 0, 0]) if abs(seg_unit[0]) < 0.9 else np.array([0, 1, 0])
    perp1 = np.cross(seg_unit, ortho)
    perp1 = perp1 / np.linalg.norm(perp1)
    perp2 = np.cross(seg_unit, perp1)

    m1 = p1 * 0.7 + p2 * 0.3
    m2 = p1 * 0.3 + p2 * 0.7

    for angle in np.linspace(0, 2 * np.pi, 12, endpoint=False):
        for dist_mult in [1.0, 1.8, 2.5, 3.5, 4.5]:
            disp = dist_mult * (np.cos(angle) * perp1 + np.sin(angle) * perp2)
            c1 = np.clip(m1 + disp, bounds_min + 0.15, bounds_max - 0.15)
            c2 = np.clip(m2 + disp, bounds_min + 0.15, bounds_max - 0.15)

            if planner.check_point_collision(c1) or planner.check_point_collision(c2):
                continue
            if (not planner.check_segment_collision(p1, c1) and
                    not planner.check_segment_collision(c1, c2) and
                    not planner.check_segment_collision(c2, p2)):
                return c1.copy(), c2.copy()

    return None


def _push_waypoint_to_safety(point, planner, bounds_min, bounds_max, obstacles):
    """将碰撞路径点推到安全位置。"""
    if not planner.check_point_collision(point):
        return point

    best_pt = None
    best_dist = np.inf

    for obs in obstacles.obstacle_list:
        c = np.array(obs.center)
        direction = point - c
        d = np.linalg.norm(direction)
        if d < 1e-6:
            direction = np.array([1, 0, 0])
            d = 1e-6
        direction = direction / d

        if obs.shape == 'sphere':
            safe_r = obs.radius + 0.5
        elif obs.shape == 'cylinder':
            safe_r = obs.radius + 0.5
        elif obs.shape == 'cuboid':
            safe_r = max(obs.size) / 2 + 0.5
        else:
            safe_r = 1.5

        for scale in [1.0, 1.3, 1.6, 2.0]:
            candidate = c + direction * safe_r * scale
            candidate = np.clip(candidate, bounds_min + 0.15, bounds_max - 0.15)
            if not planner.check_point_collision(candidate):
                dd = np.linalg.norm(candidate - point)
                if dd < best_dist:
                    best_pt = candidate.copy()
                    best_dist = dd

    if best_pt is not None:
        return best_pt
    return point


def fix_path_collisions(path, planner, space_bounds, max_rounds=8):
    """段级碰撞修复：先推离碰撞路径点，再在碰撞段中插入绕行点。"""
    best_path = np.array(path, dtype=float)
    best_collisions = len(planner.check_path_safety(best_path))
    if best_collisions == 0:
        return best_path

    bounds_min, bounds_max = np.array(space_bounds[0]), np.array(space_bounds[1])

    waypoint_fixed = False
    for i in range(1, len(best_path) - 1):
        if planner.check_point_collision(best_path[i]):
            old_pt = best_path[i].copy()
            best_path[i] = _push_waypoint_to_safety(
                best_path[i], planner, bounds_min, bounds_max, planner.obstacles)
            if not np.allclose(old_pt, best_path[i]):
                waypoint_fixed = True

    if waypoint_fixed:
        new_col = len(planner.check_path_safety(best_path))
        print(f"路径点推离修复: {best_collisions} -> {new_col} 段碰撞")
        best_collisions = new_col
        if best_collisions == 0:
            return best_path

    for round_i in range(max_rounds):
        collision_segs = planner.check_path_safety(best_path)
        if not collision_segs:
            return best_path

        print(f"碰撞修复第{round_i+1}轮: {len(collision_segs)} 段碰撞 (段: {collision_segs})")

        new_points = list(best_path)
        offset = 0
        inserted = 0
        for seg_idx in collision_segs:
            real_idx = seg_idx + offset
            if real_idx >= len(new_points) - 1:
                continue
            p1 = np.array(new_points[real_idx])
            p2 = np.array(new_points[real_idx + 1])

            bypass = _find_bypass_point(p1, p2, planner, bounds_min, bounds_max)
            if bypass is not None:
                new_points.insert(real_idx + 1, bypass)
                offset += 1
                inserted += 1
            else:
                two_pt = _find_two_point_bypass(p1, p2, planner, bounds_min, bounds_max)
                if two_pt is not None:
                    c1, c2 = two_pt
                    new_points.insert(real_idx + 1, c2)
                    new_points.insert(real_idx + 1, c1)
                    offset += 2
                    inserted += 2

        if inserted == 0:
            print("  无法找到绕行点，停止修复")
            break

        candidate_path = np.array(new_points)
        new_collisions = len(planner.check_path_safety(candidate_path))

        if new_collisions < best_collisions:
            best_path = candidate_path
            best_collisions = new_collisions
            print(f"  插入 {inserted} 个绕行点，碰撞: {best_collisions}")
        else:
            print(f"  本轮修复未改善 ({new_collisions} >= {best_collisions})，回退")
            break

    if best_collisions > 0:
        print(f"碰撞修复完成，剩余 {best_collisions} 段碰撞")
    return best_path


def main():
    # ==================== 场景配置 ====================
    SEED = None             # 随机种子: None=每次不同, 整数=可复现
    SPACE_BOUNDS = np.array([[0.0, 0.0, 0.0],
                             [6.0, 20.0, 4.0]])

    N_SPHERES   = 3         # 球体数量
    N_CYLINDERS = 2         # 圆柱体数量
    N_CUBOIDS   = 3         # 长方体数量
    DENSITY     = 'medium'  # 障碍物密度: 'low' / 'medium' / 'high'
    NUM_ON_PATH = 2         # 路径连线上放置的球体数（从 N_SPHERES 中扣除）
    SAFETY_MARGIN = 1.2     # 膨胀安全裕度
    # =================================================

    if SEED is not None:
        np.random.seed(SEED)

    evaluator = PerformanceEvaluator()
    evaluator.start_timer("clean_temp_dir")
    clean_temp_dir()
    evaluator.stop_timer("clean_temp_dir")

    start_point, goal_point = generate_random_endpoints(SPACE_BOUNDS, margin=1.0)
    print(f"起点: {np.round(start_point, 2)}")
    print(f"终点: {np.round(goal_point, 2)}")
    print(f"直线距离: {np.linalg.norm(goal_point - start_point):.2f}")

    space_boundary = [[SPACE_BOUNDS[0][i], SPACE_BOUNDS[1][i]] for i in range(3)]

    evaluator.start_timer("obstacles_generation")
    obstacles = place_obstacles(
        space_boundary, start_point, goal_point,
        n_spheres=N_SPHERES,
        n_cylinders=N_CYLINDERS,
        n_cuboids=N_CUBOIDS,
        density=DENSITY,
        num_on_path=NUM_ON_PATH,
    )
    
    evaluator.record_value("obstacles_count", len(obstacles.obstacle_list))
    evaluator.stop_timer("obstacles_generation")
    
    save_obstacles_to_file(obstacles)
    
    evaluator.start_timer("obstacles_inflation")
    safety_margin = SAFETY_MARGIN
    evaluator.stop_timer("obstacles_inflation")
    
    evaluator.start_timer("planner_initialization")
    space_size = tuple(SPACE_BOUNDS[1] - SPACE_BOUNDS[0])
    planner = FIRIPlanner(obstacles=obstacles, space_size=space_size, space_bounds=SPACE_BOUNDS)
    evaluator.stop_timer("planner_initialization")
    
    print("规划路径...")
    try:
        evaluator.start_timer("path_planning")
        final_path = planner.plan_path(
            start_point,
            goal_point,
            initial_waypoints=None,
            smoothing=True,
            max_replanning_attempts=7,
            safety_margin=safety_margin
        )
        evaluator.stop_timer("path_planning")

        # 如果 planner 未直接返回路径（返回 None/False/True），尝试从 temp 目录加载已保存的路径文件
        def try_load_saved_path():
            candidates = [
                'temp/final_path.pkl',
                'temp/smoothed_path.pkl',
                'temp/adjusted_path.pkl',
                'temp/initial_path.pkl',
                'temp/path.pkl'
            ]
            for p in candidates:
                if os.path.exists(p):
                    try:
                        with open(p, 'rb') as f:
                            data = pickle.load(f)
                        print(f"Loaded path from {p}")
                        return np.array(data)
                    except Exception:
                        continue
            return None

        # 处理 planner 返回值：可能为路径数组，也可能为布尔/None，统一处理
        if final_path is None or isinstance(final_path, (bool,)) or (isinstance(final_path, np.ndarray) and final_path.size == 0):
            loaded = try_load_saved_path()
            if loaded is not None:
                final_path = loaded
                print("使用 temp 目录中的已保存路径作为 final_path")
            else:
                print("planner 未返回路径且 temp 中无已保存路径")
                final_path = None

        # 如果得到 final_path，保存原始 final_path 便于后续分析
        if final_path is not None:
            # 确保为 numpy 数组
            final_path = np.array(final_path)
            with open('temp/final_path.pkl', 'wb') as f:
                pickle.dump(final_path, f)
            print(f"Saved raw final_path to temp/final_path.pkl, points: {len(final_path)}")

        if final_path is not None:
            evaluator.record_value("path_points_count", len(final_path))
            evaluator.record_value("path_length", calculate_path_length(final_path))

            evaluator.start_timer("collision_fixing")
            collision_segs = planner.check_path_safety(final_path)
            collisions = len(collision_segs)
            smoothed_path = final_path
            if collisions > 0:
                print(f"优化管线输出仍有 {collisions} 段碰撞，启动段级修复...")
                smoothed_path = fix_path_collisions(final_path, planner, SPACE_BOUNDS)
                collision_segs = planner.check_path_safety(smoothed_path)
                collisions = len(collision_segs)
            if collisions > 0:
                print(f"警告: 最终仍有 {collisions} 段碰撞")
            else:
                print("路径碰撞检查通过，无碰撞")
            evaluator.stop_timer("collision_fixing")
            
            evaluator.start_timer("saving_results")
            with open('temp/smoothed_path.pkl', 'wb') as f:
                pickle.dump(smoothed_path, f)
            print(f"保存路径，点数: {len(smoothed_path)}")
            
            evaluator.record_value("smoothed_path_points_count", len(smoothed_path))
            evaluator.record_value("smoothed_path_length", calculate_path_length(smoothed_path))
            evaluator.record_value("final_collisions", collisions)
            
            avg_angle = analyze_path_smoothness(smoothed_path)
            evaluator.record_value("path_smoothness", avg_angle)
            evaluator.stop_timer("saving_results")
            
            evaluator.start_timer("path_analysis")
            analyze_path_results(final_path, smoothed_path, obstacles)
            evaluator.stop_timer("path_analysis")
            
            evaluator.start_timer("matplotlib_visualization")
            visualize_results(smoothed_path, obstacles, SPACE_BOUNDS)
            evaluator.stop_timer("matplotlib_visualization")
            
            print("\nOpen3D离屏渲染...")
            evaluator.start_timer("open3d_visualization")
            try:
                o3d_ok = visualize_with_open3d(
                    smoothed_path,
                    obstacles,
                    start_point,
                    goal_point,
                    inflated_obstacles=None,
                    safety_margin=safety_margin,
                    output_path='temp/open3d_visualization.png'
                )
                if not o3d_ok:
                    print("Open3D离屏渲染不可用，已使用matplotlib保存静态图像")
            except Exception as e:
                print(f"Open3D渲染失败: {str(e)}")
            evaluator.stop_timer("open3d_visualization")

            print("\n启动Open3D交互式可视化...")
            try:
                visualize_interactive(smoothed_path, obstacles, start_point, goal_point)
            except Exception as e:
                print(f"交互式可视化失败: {str(e)}")
            evaluator.save_results()
            
            return True
        else:
            print("路径规划失败")
            evaluator.record_value("planning_success", False)
            evaluator.save_results()
            return False
    
    except Exception as e:
        print(f"规划过程出错: {str(e)}")
        evaluator.record_value("error", str(e))
        evaluator.save_results()
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    print(f"\n规划结果: {'成功' if success else '失败'}")
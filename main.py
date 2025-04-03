import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d
import time
import json
from datetime import datetime

# 导入FIRI相关模块
from firi.planning.planner import FIRIPlanner
from firi.utils.obstacle_generator import ObstacleGenerator

# 创建一个简单的Obstacle类来表示障碍物
class Obstacle:
    def __init__(self, center, radius):
        self.center = center
        self.radius = radius
        
# 创建ObstacleSet类来管理多个障碍物
class ObstacleSet:
    def __init__(self):
        self.obstacle_list = []
        
    def add_obstacle(self, center, radius):
        self.obstacle_list.append(Obstacle(center, radius))
        
    def __len__(self):
        return len(self.obstacle_list)
    
    def __iter__(self):
        return iter(self.obstacle_list)

def clean_temp_dir():
    """清理临时目录"""
    temp_dir = "temp"
    if not os.path.exists(temp_dir):
        os.makedirs(temp_dir)
    else:
        # 清理临时文件
        for file in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, file)
            if os.path.isfile(file_path):
                os.remove(file_path)
    print("Temp directory cleaned")

def place_obstacles(space_boundary, n=5):
    """在空间中放置随机障碍物，确保至少有一个在起点和终点的连线上"""
    obstacles = []
    
    # 设置起点和终点
    start = np.array([1.0, 1.0, 1.0])
    goal = np.array([9.0, 9.0, 9.0])
    
    # 计算起点到终点的方向向量
    direction = goal - start
    direction_norm = np.linalg.norm(direction)
    if direction_norm > 0:
        direction = direction / direction_norm
    
    # 在起点和终点的连线上放置一个障碍物
    # 随机选择一个位置，但确保在起点和终点之间
    t = np.random.uniform(0.3, 0.7)  # 在30%到70%的位置之间
    obstacle_pos = start + t * (goal - start)
    obstacle_radius = np.random.uniform(0.8, 1.2)  # 适当调整半径
    obstacles.append({
        'center': obstacle_pos,
        'radius': obstacle_radius
    })
    print(f"Placed obstacle on path at {obstacle_pos} with radius {obstacle_radius}")
    
    # 放置剩余的随机障碍物
    for _ in range(n - 1):
        # 生成随机位置
        x = np.random.uniform(space_boundary[0][0], space_boundary[0][1])
        y = np.random.uniform(space_boundary[1][0], space_boundary[1][1])
        z = np.random.uniform(space_boundary[2][0], space_boundary[2][1])
        
        # 生成随机半径 (0.5-1.5)
        radius = np.random.uniform(0.5, 1.5)
        
        # 创建障碍物
        obstacle = {'center': np.array([x, y, z]), 'radius': radius}
        obstacles.append(obstacle)
        print(f"Placed obstacle at {obstacle['center']} with radius {obstacle['radius']}")
    
    return obstacles

def save_obstacles_to_file(obstacles):
    """保存障碍物信息到文件"""
    with open('temp/obstacles.pkl', 'wb') as f:
        pickle.dump(obstacles, f)
    print(f"Saved {len(obstacles.obstacle_list)} obstacles to file")

def save_path(path):
    """保存路径数据到文件"""
    with open('temp/final_path.pkl', 'wb') as f:
        pickle.dump(path, f)
    print(f"Saved path with {len(path)} points to file")

def calculate_path_length(path):
    """计算路径总长度"""
    if len(path) < 2:
        return 0.0
    
    total_length = 0.0
    for i in range(len(path) - 1):
        p1 = np.array(path[i])
        p2 = np.array(path[i + 1])
        total_length += np.linalg.norm(p2 - p1)
    
    return total_length

def analyze_planning_result(path, obstacles, start, goal):
    """分析规划结果"""
    print("\n=== Planning Result Analysis ===")
    
    # 1. 检查路径是否为空
    if path is None or len(path) == 0:
        print("❌ Planning Failed: No valid path generated")
        return False
    
    # 2. 检查起点和终点
    if not np.allclose(path[0], start, atol=1e-6):
        print("❌ Planning Failed: Path start does not match the specified start point")
        return False
    if not np.allclose(path[-1], goal, atol=1e-6):
        print("❌ Planning Failed: Path end does not match the specified goal point")
        return False
    
    # 3. 计算路径长度
    path_length = calculate_path_length(path)
    print(f"Path Length: {path_length:.2f}")
    
    # 4. 检查路径平滑度
    smoothness = analyze_path_smoothness(path)
    print(f"Path Smoothness: {smoothness:.2f}")
    
    # 5. 检查碰撞
    collision_points = check_collisions(path, obstacles)
    if len(collision_points) > 0:
        print(f"❌ Planning Failed: Detected {len(collision_points)} collision points")
        return False
    
    # 6. 检查路径点数量
    if len(path) < 2:
        print("❌ Planning Failed: Insufficient path points")
        return False
    
    print("✅ Planning Successful: Path meets all requirements")
    return True

def generate_initial_waypoints(start, goal, num_waypoints=5, jitter=2.0):
    """生成初始路径点，添加扰动以绕过障碍物"""
    waypoints = [start]
    
    # 计算起点到终点的向量
    direction = goal - start
    
    # 生成中间点
    for i in range(1, num_waypoints - 1):
        t = i / (num_waypoints - 1)
        
        # 基础位置（在起点和终点连线上）
        base_point = start + t * direction
        
        # 添加垂直于主方向的随机扰动
        # 创建垂直于主方向的扰动（通过创建随机向量然后正交化）
        random_vec = np.random.randn(3)
        random_vec = random_vec - np.dot(random_vec, direction) * direction / np.dot(direction, direction)
        random_vec = random_vec / (np.linalg.norm(random_vec) + 1e-10)  # 归一化
        
        # 计算扰动大小（在路径中间点处最大）
        disturbance_factor = jitter * np.sin(np.pi * t)
        
        # 应用扰动
        waypoint = base_point + disturbance_factor * random_vec
        waypoints.append(waypoint)
    
    waypoints.append(goal)
    return np.array(waypoints)

def main():
    """
    主函数，执行路径规划并可视化结果
    """
    # 创建性能评估器
    evaluator = PerformanceEvaluator()
    
    # 清理临时目录
    evaluator.start_timer("clean_temp_dir")
    clean_temp_dir()
    evaluator.stop_timer("clean_temp_dir")
    
    # 定义起点和终点
    start_point = np.array([1.0, 1.0, 1.0])
    goal_point = np.array([9.0, 9.0, 9.0])
    
    # 定义空间边界
    space_bounds = np.array([[0.0, 0.0, 0.0], [10.0, 10.0, 10.0]])
    
    # 放置障碍物
    evaluator.start_timer("obstacles_generation")
    obstacles = ObstacleSet()
    
    # 在直线路径上放置一个障碍物
    direct_path = goal_point - start_point
    direct_path_normalized = direct_path / np.linalg.norm(direct_path)
    obstacle_point = start_point + direct_path_normalized * np.linalg.norm(direct_path) * 0.55
    obstacles.add_obstacle(obstacle_point, 1.0 + 0.1 * np.random.random())
    print(f"Placed obstacle on path at {obstacle_point} with radius {obstacles.obstacle_list[-1].radius}")
    
    # 随机放置更多障碍物
    for _ in range(15):  # 增加到15个障碍物，创建更复杂的环境
        # 生成随机位置，但避开起点和终点附近
        while True:
            pos = np.random.uniform(low=[0.0, 0.0, 0.0], high=[10.0, 10.0, 10.0])
            # 确保障碍物不会太靠近起点或终点
            if (np.linalg.norm(pos - start_point) > 1.5 and 
                np.linalg.norm(pos - goal_point) > 1.5):
                break
        
        # 随机半径，但确保足够大以形成有效障碍
        radius = 0.7 + 0.8 * np.random.random()
        obstacles.add_obstacle(pos, radius)
        print(f"Placed obstacle at {pos} with radius {radius}")
    
    evaluator.record_value("obstacles_count", len(obstacles.obstacle_list))
    evaluator.stop_timer("obstacles_generation")
    
    # 保存障碍物
    save_obstacles_to_file(obstacles)
    
    # 创建膨胀障碍物(用于规划和可视化)
    evaluator.start_timer("obstacles_inflation")
    safety_margin = 1.5  # 安全边界系数
    inflated_obstacles = create_inflated_obstacles(obstacles, safety_margin)
    evaluator.stop_timer("obstacles_inflation")
    
    # 创建FIRI规划器
    evaluator.start_timer("planner_initialization")
    planner = FIRIPlanner(obstacles=obstacles, space_size=(10, 10, 10))
    evaluator.stop_timer("planner_initialization")
    
    # 生成更多初始路径点，以提高规划成功率
    evaluator.start_timer("initial_waypoints_generation")
    num_waypoints = 15  # 增加到15个点
    initial_waypoints = []
    
    # 添加起点和终点
    initial_waypoints.append(start_point)
    
    # 生成带随机扰动的中间点
    for i in range(1, num_waypoints-1):
        t = i / (num_waypoints - 1)
        # 基础路径点（线性插值）
        base_point = start_point + t * (goal_point - start_point)
        
        # 添加随机扰动，但扰动范围降低以避免生成路径偏离太远
        jitter = 2.0 # 降低扰动范围
        # 在垂直于主方向的平面上添加扰动
        if np.random.random() > 0.5:  # 50%的概率添加更小的扰动
            jitter = 1.0
            
        # 计算扰动方向，确保扰动方向与主方向正交
        main_direction = goal_point - start_point
        main_direction = main_direction / np.linalg.norm(main_direction)
        
        # 生成两个正交于主方向的向量
        v1 = np.array([1, 0, 0])
        if abs(np.dot(v1, main_direction)) > 0.9:
            v1 = np.array([0, 1, 0])
        v1 = v1 - np.dot(v1, main_direction) * main_direction
        v1 = v1 / np.linalg.norm(v1)
        
        v2 = np.cross(main_direction, v1)
        
        # 在正交平面上添加扰动
        random_angle = 2 * np.pi * np.random.random()
        random_dist = jitter * np.random.random()
        perturb_direction = v1 * np.cos(random_angle) + v2 * np.sin(random_angle)
        perturb_vector = random_dist * perturb_direction
        
        # 最终点 = 基础点 + 扰动
        perturbed_point = base_point + perturb_vector
        
        # 确保点在空间边界内
        perturbed_point = np.clip(perturbed_point, space_bounds[0], space_bounds[1])
        
        initial_waypoints.append(perturbed_point)
    
    initial_waypoints.append(goal_point)
    initial_waypoints = np.array(initial_waypoints)
    evaluator.record_value("waypoints_count", len(initial_waypoints))
    evaluator.stop_timer("initial_waypoints_generation")
    
    print("生成带扰动的初始路径点")
    
    # 使用更安全的参数进行路径规划
    print("规划路径...")
    try:
        # 设置最大重规划次数和安全边界
        evaluator.start_timer("path_planning")
        final_path = planner.plan_path(
            start_point, 
            goal_point, 
            initial_waypoints=initial_waypoints,
            smoothing=True,
            max_replanning_attempts=7,  # 增加重规划次数
            safety_margin=safety_margin  # 使用相同的安全边界参数
        )
        evaluator.stop_timer("path_planning")
        
        if final_path is not None:
            evaluator.record_value("path_points_count", len(final_path))
            evaluator.record_value("path_length", calculate_path_length(final_path))
            print(f"Saved path with {len(final_path)} points to file")
            
            # 应用进一步平滑，使用更强的角度限制和安全性检查
            print("\n应用额外平滑以降低角度...")
            evaluator.start_timer("path_smoothing")
            smoothed_path = planner.smooth_path(
                final_path, 
                window_size=3, 
                iterations=80,  # 增加迭代次数
                max_angle=40.0,  # 降低最大角度限制
                angle_weight=0.85,  # 提高角度权重
                safety_margin=2.0  # 提高安全边界
            )
            evaluator.stop_timer("path_smoothing")
            
            # 检查平滑后的路径是否安全
            evaluator.start_timer("collision_checking")
            collisions = 0
            for i in range(len(smoothed_path)):
                for obs in obstacles.obstacle_list:
                    obs_center = np.array(obs.center)
                    dist = np.linalg.norm(smoothed_path[i] - obs_center)
                    if dist < obs.radius * 1.05:  # 添加5%的安全余量
                        collisions += 1
            evaluator.stop_timer("collision_checking")
            
            if collisions > 0:
                print(f"平滑后路径不安全（{collisions}处碰撞），尝试修复...")
                # 尝试修复碰撞点
                evaluator.start_timer("collision_fixing")
                fixed_path = smoothed_path.copy()
                fix_iterations = 0
                for _ in range(5):  # 最多尝试5次修复
                    fix_iterations += 1
                    collision_count = 0
                    for i in range(len(fixed_path)):
                        for obs in obstacles.obstacle_list:
                            obs_center = np.array(obs.center)
                            obs_radius = obs.radius * 1.1  # 10%的安全余量
                            dist = np.linalg.norm(fixed_path[i] - obs_center)
                            if dist < obs_radius:
                                collision_count += 1
                                # 移动点远离障碍物
                                direction = fixed_path[i] - obs_center
                                if np.linalg.norm(direction) > 1e-6:
                                    direction = direction / np.linalg.norm(direction)
                                    fixed_path[i] = obs_center + direction * (obs_radius + 0.3)
                    
                    if collision_count == 0:
                        print(f"碰撞修复成功！")
                        smoothed_path = fixed_path
                        break
                    else:
                        print(f"仍有{collision_count}处碰撞，继续修复...")
                
                evaluator.record_value("collision_fix_iterations", fix_iterations)
                evaluator.stop_timer("collision_fixing")
            
            # 保存平滑后的路径
            evaluator.start_timer("saving_results")
            with open('temp/smoothed_path.pkl', 'wb') as f:
                pickle.dump(smoothed_path, f)
                
            print(f"保存平滑路径，点数: {len(smoothed_path)}")
            
            # 记录最终路径信息
            evaluator.record_value("smoothed_path_points_count", len(smoothed_path))
            evaluator.record_value("smoothed_path_length", calculate_path_length(smoothed_path))
            evaluator.record_value("final_collisions", collisions)
            
            # 计算路径平滑度 (平均角度)
            avg_angle = analyze_path_smoothness(smoothed_path)
            evaluator.record_value("path_smoothness", avg_angle)
            evaluator.stop_timer("saving_results")
            
            # 分析最终路径
            evaluator.start_timer("path_analysis")
            analyze_path_results(final_path, smoothed_path, obstacles)
            evaluator.stop_timer("path_analysis")
            
            # 使用matplotlib可视化结果
            evaluator.start_timer("matplotlib_visualization")
            visualize_results(smoothed_path, obstacles, space_bounds)
            evaluator.stop_timer("matplotlib_visualization")
            
            # 使用Open3D进行可视化，包括膨胀障碍物
            print("\n使用Open3D进行路径规划可视化...")
            evaluator.start_timer("open3d_visualization")
            visualize_with_open3d(
                smoothed_path, 
                obstacles, 
                start_point, 
                goal_point, 
                inflated_obstacles=inflated_obstacles, 
                safety_margin=safety_margin
            )
            evaluator.stop_timer("open3d_visualization")
            
            # 保存性能评估结果
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

def analyze_path_results(original_path, smoothed_path, obstacles):
    """分析路径规划结果"""
    print("\n=== 规划结果分析 ===")
    
    # 计算路径长度
    def compute_path_length(path):
        length = 0
        for i in range(1, len(path)):
            length += np.linalg.norm(path[i] - path[i-1])
        return length
    
    # 计算平均角度
    def compute_avg_angle(path):
        angles = []
        for i in range(1, len(path) - 1):
            v1 = path[i] - path[i-1]
            v2 = path[i+1] - path[i]
            cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            cos_angle = np.clip(cos_angle, -1.0, 1.0)
            angle = np.arccos(cos_angle) * 180 / np.pi
            angles.append(angle)
        return np.mean(angles) if angles else 0
    
    # 检查碰撞
    def count_collisions(path, obstacles, safety_margin=1.0):
        collisions = 0
        collision_points = []
        for i, point in enumerate(path):
            for obs in obstacles.obstacle_list:
                obs_center = np.array(obs.center)
                dist = np.linalg.norm(point - obs_center)
                if dist < obs.radius * safety_margin:
                    collisions += 1
                    collision_points.append(i)
                    break
        return collisions, collision_points
    
    # 分析原始路径
    orig_length = compute_path_length(original_path)
    orig_smoothness = compute_avg_angle(original_path)
    orig_collisions, _ = count_collisions(original_path, obstacles)
    
    # 分析平滑后的路径
    smooth_length = compute_path_length(smoothed_path)
    smooth_smoothness = compute_avg_angle(smoothed_path)
    smooth_collisions, collision_indices = count_collisions(smoothed_path, obstacles)
    
    # 打印结果
    print(f"路径长度: {smooth_length:.2f}")
    print(f"路径平滑度: {smooth_smoothness:.2f}")
    
    if smooth_collisions > 0:
        print(f"❌ 规划失败: 检测到 {smooth_collisions} 个碰撞点")
    else:
        print(f"✅ 规划成功: 无碰撞")
    
    # 保存分析结果
    with open('temp/path_safety.txt', 'w') as f:
        f.write(f"Length: {smooth_length:.2f}\n")
        f.write(f"Smoothness: {smooth_smoothness:.2f}\n")
        f.write(f"Collisions: {smooth_collisions}\n")
        if collision_indices:
            f.write(f"Collision points: {collision_indices}\n")
    
    print("\nVisualizing path planning results (path & obstacles only)...")
    
    if smooth_collisions > 0:
        print("\n规划结果: 失败")
    else:
        print("\n规划结果: 成功")

def visualize_results(path, obstacles, space_bounds):
    """
    可视化路径规划结果
    
    参数:
        path: 规划路径点
        obstacles: 障碍物集合
        space_bounds: 空间边界 [[min_x, min_y, min_z], [max_x, max_y, max_z]]
    """
    try:
        # 设置matplotlib使用英文字体
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['axes.unicode_minus'] = False
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # 绘制路径
        path_array = np.array(path)
        ax.plot(path_array[:, 0], path_array[:, 1], path_array[:, 2], 'b-', linewidth=2, label='Path')
        
        # 标记起点和终点
        ax.scatter(path_array[0, 0], path_array[0, 1], path_array[0, 2], color='green', s=100, marker='o', label='Start')
        ax.scatter(path_array[-1, 0], path_array[-1, 1], path_array[-1, 2], color='red', s=100, marker='o', label='End')
        
        # 绘制障碍物
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)
        
        for i, obs in enumerate(obstacles.obstacle_list):
            center = np.array(obs.center)
            radius = obs.radius
            
            x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
            y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
            z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
            
            ax.plot_surface(x, y, z, color='r', alpha=0.3)
        
        # 设置图形属性 (使用英文标签)
        ax.set_xlabel('X-axis')
        ax.set_ylabel('Y-axis')
        ax.set_zlabel('Z-axis')
        ax.set_title('Path Planning Visualization')
        
        # 设置轴范围
        ax.set_xlim(space_bounds[0][0], space_bounds[1][0])
        ax.set_ylim(space_bounds[0][1], space_bounds[1][1])
        ax.set_zlim(space_bounds[0][2], space_bounds[1][2])
        
        # 添加图例
        ax.legend()
        
        # 保存图像
        plt.savefig('temp/path_visualization.png', dpi=300, bbox_inches='tight')
        
        print("Visualization saved to temp/path_visualization.png")
    except Exception as e:
        print(f"Visualization error: {str(e)}")

def analyze_path_smoothness(path):
    """分析路径平滑度，计算路径拐角的平均角度"""
    if len(path) < 3:
        return 0.0
    
    angles = []
    for i in range(1, len(path) - 1):
        v1 = np.array(path[i]) - np.array(path[i-1])
        v2 = np.array(path[i+1]) - np.array(path[i])
        
        # 计算向量的模长
        v1_norm = np.linalg.norm(v1)
        v2_norm = np.linalg.norm(v2)
        
        # 避免除以零
        if v1_norm < 1e-6 or v2_norm < 1e-6:
            continue
            
        # 计算夹角余弦值
        cos_angle = np.dot(v1, v2) / (v1_norm * v2_norm)
        cos_angle = np.clip(cos_angle, -1.0, 1.0)  # 避免数值误差
        
        # 转换为角度
        angle = np.arccos(cos_angle) * 180 / np.pi
        angles.append(angle)
    
    # 返回平均角度
    return np.mean(angles) if angles else 0.0

def check_collisions(path, obstacles):
    """检查路径是否与障碍物碰撞
    
    参数:
        path: 路径点列表
        obstacles: 障碍物集合
        
    返回:
        collision_points: 碰撞点列表
    """
    collision_points = []
    
    if obstacles is None:
        return collision_points
        
    # 检查每个路径点
    for i, point in enumerate(path):
        point = np.array(point)
        for obs in obstacles.obstacle_list:
            # 计算点到障碍物中心的距离
            dist = np.linalg.norm(point - np.array(obs.center))
            
            # 如果距离小于障碍物半径，则发生碰撞
            if dist <= obs.radius:
                collision_points.append(point)
                break
                
    return collision_points

def visualize_with_open3d(path, obstacles, start_point, goal_point, inflated_obstacles=None, safety_margin=1.5):
    """
    使用Open3D可视化路径规划结果，包括障碍物膨胀和轨迹
    
    参数:
        path: 规划的路径点列表
        obstacles: 原始障碍物集合
        start_point: 起点坐标
        goal_point: 终点坐标
        inflated_obstacles: 膨胀后的障碍物集合(可选)
        safety_margin: 障碍物膨胀系数(如果没有提供膨胀障碍物)
    """
    # 创建可视化器
    vis = o3d.visualization.Visualizer()
    vis.create_window(width=1200, height=800, window_name="Path Planning with Open3D")
    
    # 获取渲染选项
    opt = vis.get_render_option()
    opt.background_color = np.array([0.1, 0.1, 0.1])  # 深灰色背景
    opt.point_size = 8.0
    
    # 添加坐标系
    coord_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0)
    vis.add_geometry(coord_frame)
    
    # 添加原始障碍物(红色)
    for obs in obstacles.obstacle_list:
        center = np.array(obs.center)
        radius = obs.radius
        
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
        sphere.translate(center)
        sphere.compute_vertex_normals()
        sphere.paint_uniform_color([1, 0, 0])  # 红色
        vis.add_geometry(sphere)
    
    # 添加膨胀障碍物(半透明粉色)
    if inflated_obstacles is None and safety_margin > 1.0:
        # 如果没有提供膨胀障碍物，则根据安全系数创建
        for obs in obstacles.obstacle_list:
            center = np.array(obs.center)
            # 膨胀半径
            inflated_radius = obs.radius * safety_margin
            
            # 创建膨胀的球体
            inflated_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=inflated_radius)
            inflated_sphere.translate(center)
            inflated_sphere.compute_vertex_normals()
            # 半透明粉色
            inflated_sphere.paint_uniform_color([1, 0.7, 0.7])
            
            # 使用点云的alpha值显示为半透明
            # 注意：某些版本的Open3D可能不支持材质属性
            try:
                # 尝试设置透明度
                material = o3d.visualization.rendering.MaterialRecord()
                material.shader = "defaultLit"
                material.base_color = [1, 0.7, 0.7, 0.3]  # RGBA, alpha=0.3
                vis.add_geometry(inflated_sphere, material)
            except:
                # 如果不支持材质，则直接添加
                vis.add_geometry(inflated_sphere)
    elif inflated_obstacles is not None:
        # 如果提供了膨胀障碍物，直接使用
        for obs in inflated_obstacles.obstacle_list:
            center = np.array(obs.center)
            radius = obs.radius
            
            inflated_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=radius)
            inflated_sphere.translate(center)
            inflated_sphere.compute_vertex_normals()
            inflated_sphere.paint_uniform_color([1, 0.7, 0.7])
            vis.add_geometry(inflated_sphere)
    
    # 添加路径(加粗黄色线)
    if path is not None and len(path) > 1:
        # 创建路径线段
        line_set = o3d.geometry.LineSet()
        
        # 转换路径点为Open3D格式
        points = o3d.utility.Vector3dVector(path)
        
        # 创建线段索引
        lines = [[i, i+1] for i in range(len(path)-1)]
        line_indices = o3d.utility.Vector2iVector(lines)
        
        # 设置点和线
        line_set.points = points
        line_set.lines = line_indices
        
        # 设置线的颜色为黄色
        colors = [[1, 1, 0] for _ in range(len(lines))]
        line_set.colors = o3d.utility.Vector3dVector(colors)
        
        # 为了加粗线条，我们创建多个略有偏移的线
        vis.add_geometry(line_set)
        
        # 创建路径点云(黄色点)
        path_points = o3d.geometry.PointCloud()
        path_points.points = o3d.utility.Vector3dVector(path)
        path_points.paint_uniform_color([1, 1, 0])  # 黄色
        vis.add_geometry(path_points)
    
    # 添加起点(绿色球体)
    start_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    start_sphere.translate(start_point)
    start_sphere.paint_uniform_color([0, 1, 0])  # 绿色
    vis.add_geometry(start_sphere)
    
    # 添加终点(蓝色球体)
    end_sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.3)
    end_sphere.translate(goal_point)
    end_sphere.paint_uniform_color([0, 0, 1])  # 蓝色
    vis.add_geometry(end_sphere)
    
    # 添加轨迹起点和终点的文字标记
    # 注意：Open3D的文本渲染支持有限，这里我们使用简单的方法
    
    # 设置视图
    vis.get_view_control().set_zoom(0.7)
    
    # 尝试捕获截图
    try:
        vis.poll_events()
        vis.update_renderer()
        vis.capture_screen_image("temp/open3d_path_planning.png")
        print("Saved Open3D visualization to temp/open3d_path_planning.png")
    except Exception as e:
        print(f"Error capturing screenshot: {e}")
    
    # 运行可视化器
    vis.run()
    vis.destroy_window()
    
    return True

def create_inflated_obstacles(obstacles, safety_margin=1.5):
    """
    创建膨胀后的障碍物集合
    
    参数:
        obstacles: 原始障碍物集合
        safety_margin: 安全边界系数
    
    返回:
        膨胀后的障碍物集合
    """
    inflated_obstacles = ObstacleSet()
    
    for obs in obstacles.obstacle_list:
        # 复制中心点
        center = np.array(obs.center).copy()
        # 膨胀半径
        inflated_radius = obs.radius * safety_margin
        # 添加到膨胀障碍物集合
        inflated_obstacles.add_obstacle(center, inflated_radius)
    
    return inflated_obstacles

class PerformanceEvaluator:
    """
    Performance evaluation tool for recording algorithm execution time and resource usage
    """
    def __init__(self, output_file="temp/performance_data.json"):
        self.timestamps = {}
        self.durations = {}
        self.start_times = {}
        self.output_file = output_file
        self.system_info = self._get_system_info()
        
        # Record program start time
        self.program_start_time = time.time()
        self.timestamps["program_start"] = self._get_current_time_str()
    
    def _get_system_info(self):
        """Get system information"""
        import platform
        import os
        import psutil
        
        try:
            # Get basic system info
            system_info = {
                "platform": platform.platform(),
                "processor": platform.processor(),
                "python_version": platform.python_version(),
                "cpu_count": os.cpu_count(),
            }
            
            # Try to get detailed CPU and memory info
            try:
                import psutil
                memory = psutil.virtual_memory()
                system_info.update({
                    "total_memory_gb": round(memory.total / (1024 ** 3), 2),
                    "available_memory_gb": round(memory.available / (1024 ** 3), 2),
                })
            except ImportError:
                pass
                
            # Try to detect if running on Jetson platform
            try:
                if os.path.exists("/etc/nv_tegra_release"):
                    system_info["platform_type"] = "NVIDIA Jetson"
                    # Try to get Jetson model
                    try:
                        with open("/proc/device-tree/model", "r") as f:
                            jetson_model = f.read().strip()
                            system_info["jetson_model"] = jetson_model
                    except:
                        pass
            except:
                pass
                
            return system_info
        except Exception as e:
            print(f"Error getting system info: {e}")
            return {"error": str(e)}
    
    def _get_current_time_str(self):
        """Get formatted string of current time"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")
    
    def start_timer(self, step_name):
        """Start timing for a specific step"""
        self.start_times[step_name] = time.time()
        self.timestamps[f"{step_name}_start"] = self._get_current_time_str()
        return self.start_times[step_name]
    
    def stop_timer(self, step_name):
        """Stop timing for a specific step and record duration"""
        if step_name in self.start_times:
            end_time = time.time()
            duration = end_time - self.start_times[step_name]
            self.durations[step_name] = duration
            self.timestamps[f"{step_name}_end"] = self._get_current_time_str()
            print(f"Step '{step_name}' took: {duration:.4f} seconds")
            return duration
        else:
            print(f"Error: Step '{step_name}' was not started")
            return None
    
    def record_value(self, key, value):
        """Record a specific value (e.g., path length, point count)"""
        self.durations[key] = value
    
    def save_results(self):
        """Save performance evaluation results to file"""
        # Calculate total execution time
        total_time = time.time() - self.program_start_time
        self.durations["total_execution_time"] = total_time
        self.timestamps["program_end"] = self._get_current_time_str()
        
        # Prepare output data
        output_data = {
            "system_info": self.system_info,
            "timestamps": self.timestamps,
            "durations": self.durations,
            "total_execution_time": total_time
        }
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(self.output_file), exist_ok=True)
        
        # Save to JSON file
        with open(self.output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        # Also generate a readable text report
        report_file = self.output_file.replace('.json', '.txt')
        with open(report_file, 'w') as f:
            f.write("=== Path Planning Algorithm Performance Report ===\n\n")
            
            f.write("System Information:\n")
            for key, value in self.system_info.items():
                f.write(f"  {key}: {value}\n")
            
            f.write("\nTimestamps:\n")
            for key, value in self.timestamps.items():
                f.write(f"  {key}: {value}\n")
                
            f.write("\nExecution Times:\n")
            for key, value in self.durations.items():
                if isinstance(value, (int, float)):
                    if key.endswith("_count") or key.endswith("_size") or key.endswith("_length"):
                        f.write(f"  {key}: {value}\n")
                    else:
                        f.write(f"  {key}: {value:.4f} seconds\n")
                else:
                    f.write(f"  {key}: {value}\n")
            
            f.write(f"\nTotal Execution Time: {total_time:.4f} seconds\n")
            f.write(f"\nReport Generated: {self._get_current_time_str()}\n")
        
        print(f"Performance results saved to {self.output_file} and {report_file}")
        
        # Generate performance chart
        self._generate_performance_chart()
        
        return output_data
    
    def _generate_performance_chart(self):
        """Generate performance chart"""
        try:
            # Set font to ensure compatibility
            plt.rcParams['font.family'] = 'DejaVu Sans'
            
            # Filter timing-related data
            timing_data = {k: v for k, v in self.durations.items() 
                          if isinstance(v, (int, float)) and not (
                              k.endswith("_count") or 
                              k.endswith("_size") or 
                              k.endswith("_length") or
                              k == "total_execution_time"
                          )}
            
            if not timing_data:
                return
            
            # Sort by value
            sorted_items = sorted(timing_data.items(), key=lambda x: x[1], reverse=True)
            labels = [item[0] for item in sorted_items]
            values = [item[1] for item in sorted_items]
            
            # Create bar chart
            plt.figure(figsize=(12, 6))
            bars = plt.barh(labels, values, color='skyblue')
            
            # Add value labels
            for i, v in enumerate(values):
                plt.text(v + 0.01, i, f"{v:.4f}s", va='center')
            
            plt.title('Algorithm Stage Execution Times (seconds)')
            plt.xlabel('Execution Time (seconds)')
            plt.tight_layout()
            
            # Save chart
            chart_path = self.output_file.replace('.json', '_chart.png')
            plt.savefig(chart_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            print(f"Performance chart saved to {chart_path}")
        except Exception as e:
            print(f"Error generating performance chart: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    success = main()
    print(f"\n规划结果: {'成功' if success else '失败'}") 
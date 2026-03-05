import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.interpolate import splprep, splev

def visualize_results(path, obstacles, space_bounds):
    """
    使用matplotlib可视化路径规划结果

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

        # 使用B样条函数平滑路径
        tck, u = splprep(path.T, u=None, s=0.0, k=3)  # s=0 表示不平滑，k=3 表示三次样条
        u_new = np.linspace(u.min(), u.max(), 1000)  # 生成新的参数值
        path_smooth = np.array(splev(u_new, tck)).T  # 计算平滑后的路径点

        # 绘制平滑后的路径
        ax.plot(path_smooth[:, 0], path_smooth[:, 1], path_smooth[:, 2], 'b-', linewidth=2, label='Smoothed Path')

        # 标记起点和终点
        ax.scatter(path[0, 0], path[0, 1], path[0, 2], color='green', s=100, marker='o', label='Start')
        ax.scatter(path[-1, 0], path[-1, 1], path[-1, 2], color='red', s=100, marker='o', label='End')

        # 绘制障碍物
        u = np.linspace(0, 2 * np.pi, 20)
        v = np.linspace(0, np.pi, 10)

        for obs in obstacles:
            center = np.array(obs.center)
            print(f"Rendering obstacle with shape: {obs.shape}")  # 调试输出障碍物的形状

            if obs.shape == 'sphere':
                radius = obs.radius
                # 使用 parametric 方程绘制球体
                x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
                y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
                z = center[2] + radius * np.outer(np.ones(np.size(u)), np.cos(v))
                ax.plot_surface(x, y, z, color='r', alpha=0.3)

            elif obs.shape == 'cylinder':
                radius = obs.radius
                height = obs.height
                # 绘制圆柱体，使用圆柱体的顶端和底端圆形网格
                z = np.linspace(center[2] - height / 2, center[2] + height / 2, 20)
                theta = np.linspace(0, 2 * np.pi, 30)
                theta_grid, z_grid = np.meshgrid(theta, z)
                x_grid = radius * np.cos(theta_grid) + center[0]
                y_grid = radius * np.sin(theta_grid) + center[1]
                ax.plot_surface(x_grid, y_grid, z_grid, color='b', alpha=0.3)

            elif obs.shape == 'cuboid':
                # 使用 Poly3DCollection 绘制长方体
                length, width, height = obs.size
                # 定义长方体的8个顶点
                vertices = np.array([
                    [center[0] - length / 2, center[1] - width / 2, center[2] - height / 2],
                    [center[0] + length / 2, center[1] - width / 2, center[2] - height / 2],
                    [center[0] + length / 2, center[1] + width / 2, center[2] - height / 2],
                    [center[0] - length / 2, center[1] + width / 2, center[2] - height / 2],
                    [center[0] - length / 2, center[1] - width / 2, center[2] + height / 2],
                    [center[0] + length / 2, center[1] - width / 2, center[2] + height / 2],
                    [center[0] + length / 2, center[1] + width / 2, center[2] + height / 2],
                    [center[0] - length / 2, center[1] + width / 2, center[2] + height / 2]
                ])

                # 定义长方体的面（每个面由四个顶点组成）
                faces = [
                    [vertices[0], vertices[1], vertices[2], vertices[3]],  # 底面
                    [vertices[4], vertices[5], vertices[6], vertices[7]],  # 顶面
                    [vertices[0], vertices[1], vertices[5], vertices[4]],  # 前面
                    [vertices[1], vertices[2], vertices[6], vertices[5]],  # 右面
                    [vertices[2], vertices[3], vertices[7], vertices[6]],  # 后面
                    [vertices[3], vertices[0], vertices[4], vertices[7]]   # 左面
                ]
                
                # 使用 Poly3DCollection 绘制这些面
                ax.add_collection3d(Poly3DCollection(faces, facecolors='g', linewidths=1, edgecolors='r', alpha=0.3))

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

class _SuppressNativeOutput:
    """OS 级别同时抑制 stdout/stderr (C 层 libGL + Open3D WARNING)"""
    def __enter__(self):
        import os, sys
        self._stderr_fd = os.dup(2)
        self._stdout_fd = os.dup(1)
        self._devnull = os.open(os.devnull, os.O_WRONLY)
        os.dup2(self._devnull, 2)
        os.dup2(self._devnull, 1)
        self._py_stderr = sys.stderr
        self._py_stdout = sys.stdout
        sys.stderr = open(os.devnull, 'w')
        sys.stdout = open(os.devnull, 'w')
        return self

    def __exit__(self, *args):
        import os, sys
        sys.stderr.close()
        sys.stdout.close()
        sys.stderr = self._py_stderr
        sys.stdout = self._py_stdout
        os.dup2(self._stderr_fd, 2)
        os.dup2(self._stdout_fd, 1)
        os.close(self._stderr_fd)
        os.close(self._stdout_fd)
        os.close(self._devnull)


def visualize_with_open3d(path, obstacles, start_point, goal_point,
                          inflated_obstacles=None, safety_margin=1.2,
                          output_path='temp/open3d_visualization.png',
                          width=1600, height=1200):
    """
    使用 Open3D OffscreenRenderer 离屏渲染路径规划结果并保存为图片。
    无需 DISPLAY，使用 EGL headless 模式。
    返回 True 表示成功，False 表示失败。
    """
    with _SuppressNativeOutput():
        try:
            import open3d as o3d
            import open3d.visualization.rendering as rendering
        except ImportError:
            pass
    try:
        o3d, rendering
    except NameError:
        print("Open3D: 未安装open3d库，跳过可视化")
        return False

    try:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    except Exception:
        pass

    try:
        renderer = rendering.OffscreenRenderer(width, height)
    except Exception as e:
        print(f"Open3D: OffscreenRenderer 创建失败: {e}")
        return False

    renderer.scene.set_background([0.12, 0.12, 0.15, 1.0])

    mat_red = rendering.MaterialRecord()
    mat_red.shader = 'defaultLit'
    mat_red.base_color = [0.9, 0.2, 0.15, 0.85]

    mat_green = rendering.MaterialRecord()
    mat_green.shader = 'defaultLit'
    mat_green.base_color = [0.2, 0.8, 0.3, 0.85]

    mat_blue = rendering.MaterialRecord()
    mat_blue.shader = 'defaultLit'
    mat_blue.base_color = [0.2, 0.4, 0.9, 0.85]

    mat_start = rendering.MaterialRecord()
    mat_start.shader = 'defaultLit'
    mat_start.base_color = [0.1, 1.0, 0.2, 1.0]

    mat_goal = rendering.MaterialRecord()
    mat_goal.shader = 'defaultLit'
    mat_goal.base_color = [1.0, 0.3, 0.1, 1.0]

    mat_path = rendering.MaterialRecord()
    mat_path.shader = 'defaultUnlit'
    mat_path.base_color = [1.0, 0.85, 0.0, 1.0]
    mat_path.line_width = 3.0

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8)
    renderer.scene.add_geometry('coord', coord, rendering.MaterialRecord())

    for idx, obs in enumerate(obstacles.obstacle_list):
        center = np.array(obs.center)
        if obs.shape == 'sphere':
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=obs.radius, resolution=20)
            mesh.translate(center)
            mesh.compute_vertex_normals()
            renderer.scene.add_geometry(f'obs_sphere_{idx}', mesh, mat_red)
        elif obs.shape == 'cylinder':
            mesh = o3d.geometry.TriangleMesh.create_cylinder(obs.radius, obs.height, resolution=20)
            mesh.translate(center)
            mesh.compute_vertex_normals()
            renderer.scene.add_geometry(f'obs_cyl_{idx}', mesh, mat_green)
        elif obs.shape == 'cuboid':
            l, w, h = obs.size
            mesh = o3d.geometry.TriangleMesh.create_box(l, w, h)
            mesh.translate(center - np.array([l/2, w/2, h/2]))
            mesh.compute_vertex_normals()
            renderer.scene.add_geometry(f'obs_box_{idx}', mesh, mat_blue)

    if path is not None and len(path) >= 4:
        path_array = np.array(path)
        try:
            tck, u = splprep(path_array.T, u=None, s=0.0, k=3)
            u_new = np.linspace(u.min(), u.max(), 500)
            path_smooth = np.array(splev(u_new, tck)).T

            line_set = o3d.geometry.LineSet()
            line_set.points = o3d.utility.Vector3dVector(path_smooth)
            line_set.lines = o3d.utility.Vector2iVector(
                [[i, i+1] for i in range(len(path_smooth)-1)])
            line_set.colors = o3d.utility.Vector3dVector(
                [[1.0, 0.85, 0.0]] * (len(path_smooth)-1))
            renderer.scene.add_geometry('path_line', line_set, mat_path)
        except Exception as e:
            print(f"B样条曲线生成失败: {e}")

    sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.25, resolution=16)
    sp.translate(start_point)
    sp.compute_vertex_normals()
    renderer.scene.add_geometry('start', sp, mat_start)

    gp = o3d.geometry.TriangleMesh.create_sphere(radius=0.25, resolution=16)
    gp.translate(goal_point)
    gp.compute_vertex_normals()
    renderer.scene.add_geometry('goal', gp, mat_goal)

    scene_center = (np.array(start_point) + np.array(goal_point)) / 2
    dist = np.linalg.norm(np.array(goal_point) - np.array(start_point))
    eye = scene_center + np.array([dist * 0.6, -dist * 0.3, dist * 0.5])
    renderer.setup_camera(45.0, scene_center, eye, [0, 0, 1])

    img = renderer.render_to_image()
    o3d.io.write_image(output_path, img)
    print(f"Open3D离屏渲染已保存: {output_path} ({width}x{height})")
    return True


def visualize_interactive(path, obstacles, start_point, goal_point):
    """
    Open3D 交互式 3D 可视化（需要有显示设备）。
    窗口打开后可用鼠标旋转/缩放，关闭窗口后程序继续。
    返回 True 表示成功。
    """
    with _SuppressNativeOutput():
        try:
            import open3d as o3d
        except ImportError:
            pass
    try:
        o3d
    except NameError:
        print("Open3D: 未安装，跳过交互式可视化")
        return False

    try:
        o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Warning)
    except Exception:
        pass

    vis = o3d.visualization.Visualizer()
    with _SuppressNativeOutput():
        try:
            created = vis.create_window(width=1400, height=900, window_name="FIRI Path Planning")
        except Exception:
            created = False
    if not created:
        print("Open3D: 无法创建交互窗口")
        return False

    opt = vis.get_render_option()
    if opt is not None:
        opt.background_color = np.array([0.1, 0.1, 0.12])
        opt.point_size = 5.0
        opt.line_width = 3.0

    coord = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.8)
    vis.add_geometry(coord)

    for obs in obstacles.obstacle_list:
        center = np.array(obs.center)
        if obs.shape == 'sphere':
            mesh = o3d.geometry.TriangleMesh.create_sphere(radius=obs.radius, resolution=20)
            mesh.translate(center)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.9, 0.2, 0.15])
            vis.add_geometry(mesh)
        elif obs.shape == 'cylinder':
            mesh = o3d.geometry.TriangleMesh.create_cylinder(obs.radius, obs.height, resolution=20)
            mesh.translate(center)
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.2, 0.8, 0.3])
            vis.add_geometry(mesh)
        elif obs.shape == 'cuboid':
            l, w, h = obs.size
            mesh = o3d.geometry.TriangleMesh.create_box(l, w, h)
            mesh.translate(center - np.array([l/2, w/2, h/2]))
            mesh.compute_vertex_normals()
            mesh.paint_uniform_color([0.2, 0.4, 0.9])
            vis.add_geometry(mesh)

    if path is not None and len(path) >= 4:
        path_array = np.array(path)
        try:
            tck, u = splprep(path_array.T, u=None, s=0.0, k=3)
            u_new = np.linspace(u.min(), u.max(), 500)
            path_smooth = np.array(splev(u_new, tck)).T
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(path_smooth)
            ls.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(path_smooth)-1)])
            ls.colors = o3d.utility.Vector3dVector([[1.0, 0.85, 0.0]] * (len(path_smooth)-1))
            vis.add_geometry(ls)
        except Exception:
            ls = o3d.geometry.LineSet()
            ls.points = o3d.utility.Vector3dVector(path_array)
            ls.lines = o3d.utility.Vector2iVector([[i, i+1] for i in range(len(path_array)-1)])
            ls.colors = o3d.utility.Vector3dVector([[1, 1, 0]] * (len(path_array)-1))
            vis.add_geometry(ls)

    sp = o3d.geometry.TriangleMesh.create_sphere(radius=0.3, resolution=16)
    sp.translate(start_point)
    sp.compute_vertex_normals()
    sp.paint_uniform_color([0.1, 1.0, 0.2])
    vis.add_geometry(sp)

    gp = o3d.geometry.TriangleMesh.create_sphere(radius=0.3, resolution=16)
    gp.translate(goal_point)
    gp.compute_vertex_normals()
    gp.paint_uniform_color([1.0, 0.3, 0.1])
    vis.add_geometry(gp)

    print("Open3D交互式窗口已打开 (关闭窗口后程序继续)")
    vis.run()
    vis.destroy_window()
    return True

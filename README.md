# MVIE-ConvexPlanner：基于最大体积内接椭球的凸走廊路径规划

<p align="center">
  <img src="MVIE-ConvexPlanner.jpg" width="600" alt="MVIE-ConvexPlanner Algorithm"/>
</p>

## 项目介绍

MVIE-ConvexPlanner 是一种面向三维障碍环境的安全轨迹规划算法。该算法在浙江大学 FastLab 的 [FIRI 算法](https://ieeexplore.ieee.org/document/9697174)基础上进行改进，通过**迭代安全推离**、**MVIE 凸走廊**和**约束轨迹优化**三个阶段，生成动力学可行且无碰撞的平滑轨迹。

> **FIRI 原文**: *J. Liu, et al., "Fast Iterative Region Inflation for Computing Large 2-D/3-D Convex Regions of Obstacle-Free Space," IEEE RA-L, 2022.*

## 相对于 FIRI 的改进

| 改进项 | FIRI 原版 | MVIE-ConvexPlanner |
|--------|-----------|-------------------|
| 安全推离 | 无 | FIRI 前迭代推离控制点至安全空间 |
| 走廊约束 | 启发式路径点调整 | 椭球体走廊作为优化硬约束 |
| 轨迹优化 | 无约束 B-spline 平滑 | SLSQP 约束优化 (a_max, jerk_max) |
| 障碍物类型 | 仅球体 | 球体、圆柱体、长方体、任意朝向有限胶囊体树枝 |
| 碰撞修复 | 无 | 段级绕行点插入 + 多点绕行 |
| 碰撞消除率 | ~85% | **100%** (30场景测试) |

## 算法流程

算法按 Algorithm 1 (MVIE-ConvexPlanner) 伪代码执行：

1. **点云/障碍物预处理** — 构建 KD-Tree 加速近邻查询
2. **初始控制点生成** — 起终点间线性骨架 + 正弦扰动
3. **迭代安全推离** (Steps 5-13) — 将不安全控制点推离障碍物
4. **安全走廊计算** (Steps 15-22) — 对每段路径做 FIRI 膨胀 + MVIE 求解
5. **约束轨迹优化** (Step 25) — SLSQP 优化控制点，约束：走廊内、加速度/jerk 限制
6. **B-spline 平滑** — 输出动力学可行的三次 B-spline 轨迹

## 网络架构 / 核心模块

```
MVIE-ConvexPlanner/
├── main.py                         # 主程序入口（场景配置、规划流程）
├── obstacle_generator.py           # 障碍物生成（球/柱/长方体, 密度控制）
├── visualizer.py                   # 可视化（Matplotlib + Open3D）
├── performance_evaluator.py        # 性能评估器
├── path_planner.py                 # 基础路径工具
├── utils.py                        # 通用工具函数
├── firi/                           # 核心算法包
│   ├── geometry/
│   │   ├── ellipsoid.py            # 椭球体（SVD分解、半空间变换）
│   │   └── convex_polytope.py      # 凸多胞体（半空间/顶点表示、Chebyshev中心）
│   ├── planning/
│   │   ├── config.py               # 配置管理（d_safe, a_max, jerk_max 等）
│   │   ├── firi.py                 # FIRI 核心（restrictive_inflation）
│   │   ├── mvie.py                 # MVIE 求解（Affine Scaling + Khachiyan）
│   │   ├── plannerv2.py            # 主规划器（安全推离/走廊/优化/重规划）
│   │   └── planner.py              # 原始规划器（兼容保留）
│   └── utils/
│       ├── obstacle_generator.py   # 内部障碍物工具
│       └── analyze_path.py         # 路径分析
├── test/                           # 测试脚本
├── temp/                           # 运行时输出（自动生成）
└── CHANGELOG.md                    # 变更日志
```

## 安装

### 环境要求

- Python 3.8+
- NumPy, SciPy, Matplotlib
- Open3D（可选，用于交互式 3D 可视化）

### 安装步骤

```bash
git clone https://github.com/snow-wind-001/MVIE-ConvexPlanner.git
cd MVIE-ConvexPlanner

pip install numpy scipy matplotlib open3d psutil
```

## 使用方法

### 运行路径规划

```bash
python main.py
```

在 `main.py` 顶部可配置：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `SEED` | int/None | None | 随机种子，None 为每次不同 |
| `SPACE_BOUNDS` | ndarray | [[0,0,0],[6,20,4]] | 仿真空间边界 |
| `N_SPHERES` | int | 3 | 球体障碍物数量 |
| `N_CYLINDERS` | int | 2 | 圆柱体数量 |
| `N_CUBOIDS` | int | 3 | 长方体数量 |
| `DENSITY` | str | 'medium' | 障碍物密度: 'low'/'medium'/'high' |
| `NUM_ON_PATH` | int | 2 | 路径上放置的球体数 |
| `SAFETY_MARGIN` | float | 0.30 | 障碍物表面的绝对安全净空（米） |
| `PLANNING_MODE` | str | 'realtime' | `realtime` 50 Hz局部层；`full` FIRI/MVIE全局层 |
| `REALTIME_BUDGET` | float | 0.020 | 实时局部层时间预算（秒） |
| `ENABLE_VISUALIZATION` | bool | False | 实时部署关闭；仿真出图时开启 |

### 双频实时架构

- `realtime`：使用 `17×13` 球面深度图、距离相关安全膨胀和有界候选验证；显式保留左、右、上、下候选扇区，超时或无安全解时返回失败。
- `full`：运行安全推离、FIRI/MVIE、多胞体约束 SLSQP 和安全后处理，用于启动阶段或环境显著变化后的低频全局重规划。
- 工程部署时应缓存 `full` 输出作为 `plan_realtime(..., reference_path=...)` 的参考路径；控制循环只运行实时局部层。

### 轨迹分析

```bash
python analyze_trajectory.py   # 分析路径角度、曲率、安全性
python angle_comparison.py     # 原始 vs 平滑路径对比
```

### Three.js 无人机穿林测试

`forest_simulator/` 提供可交互的 Three.js 测试场景。浏览器仅负责渲染；
场景 JSON 中的路径、耗时、碰撞状态和最小净空均由 Python
`FIRIPlanner` 生成。

```bash
cd forest_simulator
npm install
npm run generate:data   # 运行 30 个固定种子并更新测试证据
npm run dev             # http://127.0.0.1:5173
```

测试协议为：full FIRI 低频生成树林参考路径；树干、三条斜向树枝和多球
树冠都参与真实三维碰撞；随后在航路中段加入新感知横穿树枝或树木，由
realtime 层在 20 ms 预算内执行一次局部修复。失败时保持悬停，不会将不安全
路径发送给控制器。

## 关键参数

在 `firi/planning/config.py` 中配置：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `d_safe` | 0.5 | 安全推离距离阈值 |
| `push_iterations` | 10 | 推离最大迭代次数 |
| `a_max` | 4.0 | 最大加速度约束（控制点二阶差分） |
| `jerk_max` | 8.0 | 最大 jerk 约束（控制点三阶差分） |
| `safety_iterations` | 2 | FIRI 迭代次数 |
| `volume_threshold` | 0.01 | MVIE 收敛阈值 |

## 性能

Three.js 固定回归集和未见种子测试结果：

| 指标 | 数值 |
|------|------|
| 固定三维树林 | **30/30** 安全成功，P95 `13.89 ms`，最大 `18.40 ms` |
| 未见三维树林 | 在 143 个 full-safe 参考中 **134/143** 安全成功 |
| 未见实时性 | **143/143** 在 `20 ms` 内结束，P95 `17.85 ms` |
| 不安全输出 | **0**（无解时返回 `None` / 安全悬停） |

## 可视化

提供两种可视化方式：
- **Matplotlib** — 静态 3D 路径图（自动保存至 `temp/path_visualization.png`）
- **Open3D** — 交互式 3D 可视化 + 离屏渲染

## 致谢

- 本项目基于浙江大学 FastLab 的 FIRI 算法进行改进
- 原始论文：Liu J, et al. *Fast Iterative Region Inflation for Computing Large 2-D/3-D Convex Regions of Obstacle-Free Space*. IEEE RA-L, 2022.

## 许可证

MIT License

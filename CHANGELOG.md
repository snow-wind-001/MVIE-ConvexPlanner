# CHANGELOG

## 2026-03-05 — 伪代码对齐：补全缺失算法步骤

### 新增功能

- **Pre-FIRI 安全推离** (`_safety_push`): 在安全走廊计算前，迭代将控制点推离障碍物 (对应伪代码 Steps 5-13)
- **安全走廊计算** (`_compute_corridors`): 基于修正后控制点计算 FIRI+MVIE 安全椭球体走廊 (Steps 15-22)
- **约束轨迹优化** (`_optimize_trajectory`): 以走廊约束 + 加速度/jerk 限制做 SLSQP 优化 (Step 25)
- 新增配置参数: `d_safe`, `push_iterations`, `push_delta`, `a_max`, `jerk_max`, `opt_max_iter`

### 改进

- 重构 `plan_path()` 按伪代码流水线执行: 推离 → 走廊 → 优化 → 平滑 → 验证
- 保留旧版启发式重规划作为 fallback

### 测试

- 30 个随机场景批量测试: 碰撞率从 6.7% 降至 0%
- 平均规划时间 2.44s，最大 7.23s（满足 60s 边缘设备限制）

### 文件变更

- `firi/planning/config.py`: 新增 6 个配置参数
- `firi/planning/plannerv2.py`: 新增 3 个方法，重构 `plan_path`
- `main.py`: 简化后处理逻辑
- `chang.log`: 记录差异分析和修复详情

---

## 2026-03-04 — 碰撞修复与段级绕行

- 修复 `check_point_collision` 符号错误 (`dist < -safe_distance` → `dist < safe_distance`)
- 实现段级碰撞修复 (`fix_path_collisions`): 插入绕行点避开障碍物
- 支持单点/双点绕行 + 路径点推离
- 添加 Open3D 交互式可视化模式

## 2026-03-04 — 功能增强

- 可配置障碍物数量、密度、类型
- 随机起终点生成（仿真空间两端）
- Open3D 离屏渲染 + 交互式可视化
- 修复 libstdc++ 兼容性问题

## 2026-03-04 — 初始修复

- 修复 numpy 数组真值判断、变量作用域、递归深度问题
- 修复 MVIE 数值稳定性（sqrt 负值、条件数退化）
- 修复空间边界约束传递
- 性能优化：总执行时间从卡死降至 < 10s

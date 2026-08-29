import contextlib
import io
import time

import numpy as np

from firi.planning.plannerv2 import FIRIPlanner
from obstacle_generator import ObstacleSet


def make_planner():
    obstacles = ObstacleSet()
    obstacles.add_obstacle("sphere", np.zeros(3), radius=1.0)
    bounds = np.array([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]])
    return FIRIPlanner(
        obstacles,
        space_size=(10.0, 10.0, 10.0),
        space_bounds=bounds,
    )


def test_safety_margin_changes_collision_decision():
    planner = make_planner()
    point = np.array([1.4, 0.0, 0.0])

    planner.set_safety_margin(0.3)
    assert not planner.check_point_collision(point)

    planner.set_safety_margin(0.5)
    assert planner.check_point_collision(point)


def test_projection_guide_uses_same_safety_margin_as_planner():
    planner = make_planner()
    planner.set_safety_margin(0.42)

    assert np.isclose(planner.spherical_guide.config.safety_radius, 0.42)
    assert np.isclose(planner.firi.safety_margin, 0.42)


def test_realtime_path_is_safe_and_meets_declared_deadline():
    planner = make_planner()
    planner.set_safety_margin(0.3)
    start = np.array([0.0, -4.0, 0.0])
    goal = np.array([0.0, 4.0, 0.0])

    path = planner.plan_realtime(
        start,
        goal,
        reference_path=np.vstack([start, goal]),
        time_budget=0.020,
    )

    assert path is not None
    assert not planner.check_path_safety(path)
    assert planner.last_realtime_stats["deadline_met"]


def test_realtime_does_not_return_a_late_path():
    planner = make_planner()
    planner.set_safety_margin(0.3)
    start = np.array([0.0, -4.0, 0.0])
    goal = np.array([0.0, 4.0, 0.0])

    class SlowGuide:
        @staticmethod
        def find_bypass(*args, **kwargs):
            time.sleep(0.015)
            return np.array([2.0, 0.0, 0.0])

    planner.realtime_guide = SlowGuide()
    path = planner.plan_realtime(
        start,
        goal,
        reference_path=np.vstack([start, goal]),
        time_budget=0.010,
    )

    assert path is None
    assert not planner.last_realtime_stats["deadline_met"]
    assert not planner.last_realtime_stats["success"]


def test_batched_path_collision_matches_segment_checks_for_native_shapes():
    obstacles = ObstacleSet()
    obstacles.add_obstacle("sphere", np.array([0.0, 0.0, 0.0]), radius=1.0)
    obstacles.add_obstacle(
        "cylinder", np.array([3.0, 0.0, 0.0]), radius=0.5, height=2.0
    )
    obstacles.add_obstacle(
        "cuboid", np.array([-3.0, 0.0, 0.0]), size=np.ones(3)
    )
    bounds = np.array([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]])
    planner = FIRIPlanner(obstacles, (10.0, 10.0, 10.0), bounds)
    planner.set_safety_margin(0.3)
    path = np.array(
        [
            [-4.0, 0.0, 0.0],
            [-2.0, 0.0, 0.0],
            [0.0, 2.0, 0.0],
            [2.0, 2.0, 0.0],
            [3.0, 0.0, 0.0],
        ]
    )

    segmentwise = [
        index
        for index in range(len(path) - 1)
        if planner.check_segment_collision(path[index], path[index + 1])
    ]
    assert planner.check_path_safety(path) == segmentwise
    assert segmentwise == [0, 3]


def test_analytic_sphere_and_in_height_cylinder_segment_checks():
    obstacles = ObstacleSet()
    obstacles.add_obstacle("sphere", np.array([-3.0, 0.0, 0.0]), radius=1.0)
    obstacles.add_obstacle(
        "cylinder", np.array([3.0, 0.0, 0.0]), radius=1.0, height=4.0
    )
    bounds = np.array([[-6.0, -6.0, -6.0], [6.0, 6.0, 6.0]])
    planner = FIRIPlanner(obstacles, (12.0, 12.0, 12.0), bounds)
    planner.set_safety_margin(0.3)

    assert planner.check_segment_collision(
        np.array([-5.0, 1.29, 0.0]), np.array([-1.0, 1.29, 0.0])
    )
    assert planner.check_segment_collision(
        np.array([1.0, 1.29, 0.0]), np.array([5.0, 1.29, 0.0])
    )
    assert not planner.check_segment_collision(
        np.array([1.0, 1.31, 0.0]), np.array([5.0, 1.31, 0.0])
    )


def test_capsule_branch_uses_exact_point_and_segment_collision():
    obstacles = ObstacleSet()
    obstacles.add_obstacle(
        "capsule",
        radius=0.2,
        start=np.array([-1.0, 0.0, 0.0]),
        end=np.array([1.0, 0.0, 0.0]),
    )
    bounds = np.array([[-4.0, -4.0, -4.0], [4.0, 4.0, 4.0]])
    planner = FIRIPlanner(obstacles, (8.0, 8.0, 8.0), bounds)
    planner.set_safety_margin(0.3)

    assert planner.check_point_collision(np.array([0.0, 0.0, 0.49]))
    assert not planner.check_point_collision(np.array([0.0, 0.0, 0.51]))
    assert planner.check_segment_collision(
        np.array([0.0, -2.0, 0.49]), np.array([0.0, 2.0, 0.49])
    )
    assert not planner.check_segment_collision(
        np.array([0.0, -2.0, 0.51]), np.array([0.0, 2.0, 0.51])
    )
    path = np.array(
        [[0.0, -2.0, 0.51], [0.0, 2.0, 0.51], [3.0, 2.0, 0.0]]
    )
    assert planner.check_path_safety(path) == []


def test_realtime_branch_bypass_can_use_vertical_motion():
    obstacles = ObstacleSet()
    obstacles.add_obstacle(
        "capsule",
        radius=0.25,
        start=np.array([-2.0, 5.0, 0.0]),
        end=np.array([2.0, 5.0, 0.0]),
    )
    bounds = np.array([[-3.0, -1.0, -3.0], [3.0, 11.0, 3.0]])
    planner = FIRIPlanner(obstacles, (6.0, 12.0, 6.0), bounds)
    planner.set_safety_margin(0.3)
    start = np.array([0.0, 0.0, 0.0])
    goal = np.array([0.0, 10.0, 0.0])

    path = planner.plan_realtime(
        start,
        goal,
        reference_path=np.vstack([start, goal]),
        time_budget=0.100,
    )

    assert path is not None
    assert not planner.check_path_safety(path)
    assert np.max(np.abs(path[:, 2])) > 0.55


def test_cylinder_cap_region_keeps_sampled_distance_fallback():
    obstacles = ObstacleSet()
    obstacles.add_obstacle(
        "cylinder", np.array([0.0, 0.0, 0.0]), radius=1.0, height=4.0
    )
    bounds = np.array([[-5.0, -5.0, -5.0], [5.0, 5.0, 5.0]])
    planner = FIRIPlanner(obstacles, (10.0, 10.0, 10.0), bounds)
    planner.set_safety_margin(0.3)

    assert planner.check_segment_collision(
        np.array([-2.0, 0.0, 2.2]), np.array([2.0, 0.0, 2.2])
    )
    assert not planner.check_segment_collision(
        np.array([-2.0, 0.0, 2.31]), np.array([2.0, 0.0, 2.31])
    )


def test_full_optimizer_converges_and_collision_state_matches_returned_path():
    bounds = np.array([[0.0, 0.0, 0.0], [6.0, 20.0, 4.0]])
    planner = FIRIPlanner(
        ObstacleSet(), tuple(bounds[1] - bounds[0]), bounds
    )
    np.random.seed(1)
    with contextlib.redirect_stdout(io.StringIO()):
        path = planner.plan_path(
            np.array([1.0, 1.0, 1.0]),
            np.array([5.0, 18.0, 3.0]),
            smoothing=False,
            safety_margin=0.3,
        )

    assert planner.last_optimization_result.success
    assert planner.last_optimization_feasible
    assert planner.path_collisions == planner.check_path_safety(path)

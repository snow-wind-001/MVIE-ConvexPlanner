import unittest

import numpy as np

from firi.planning.spherical_projection import (
    SphericalProjectionConfig,
    SphericalProjectionGuide,
)
from obstacle_generator import ObstacleSet


class SphericalProjectionTests(unittest.TestCase):
    def test_point_projection_keeps_nearest_depth_and_orientation(self):
        config = SphericalProjectionConfig(
            width=81,
            height=41,
            fov_h=np.deg2rad(90.0),
            fov_v=np.deg2rad(60.0),
            near=0.1,
            far=20.0,
            safety_radius=0.3,
        )
        guide = SphericalProjectionGuide([], config=config)
        points = np.array(
            [
                [0.0, 8.0, 0.0],
                [0.0, 5.0, 0.0],
                [1.0, 5.0, 0.0],
                [0.0, 5.0, 1.0],
            ]
        )

        projection = guide.project_points(
            points,
            origin=np.zeros(3),
            forward=np.array([0.0, 1.0, 0.0]),
        )

        center_v = config.height // 2
        center_u = config.width // 2
        self.assertTrue(
            np.isclose(projection.depth_map[center_v, center_u], 5.0)
        )

        finite_v, finite_u = np.nonzero(np.isfinite(projection.depth_map))
        self.assertTrue(np.any(finite_u > center_u))  # +world-x is camera-right
        self.assertTrue(np.any(finite_v < center_v))  # +world-z maps image-up

    def test_distance_dependent_inflation_is_larger_for_near_obstacles(self):
        config = SphericalProjectionConfig(
            width=81,
            height=41,
            fov_h=np.deg2rad(90.0),
            fov_v=np.deg2rad(60.0),
            safety_radius=0.5,
        )
        guide = SphericalProjectionGuide([], config=config)
        center = (config.height // 2, config.width // 2)

        near_depth = np.full((config.height, config.width), np.inf)
        far_depth = near_depth.copy()
        near_depth[center] = 2.0
        far_depth[center] = 8.0

        self.assertGreater(
            guide.inflate_depth_map(near_depth).sum(),
            guide.inflate_depth_map(far_depth).sum(),
        )

    def test_spherical_guide_finds_verified_bypass_for_blocked_segment(self):
        obstacles = ObstacleSet()
        obstacles.add_obstacle(
            "sphere", np.array([0.0, 5.0, 0.0]), radius=1.0
        )
        config = SphericalProjectionConfig(
            width=101,
            height=61,
            fov_h=np.deg2rad(140.0),
            fov_v=np.deg2rad(100.0),
            near=0.1,
            far=15.0,
            safety_radius=0.3,
        )
        guide = SphericalProjectionGuide(obstacles, config=config)
        start = np.array([0.0, 0.0, 0.0])
        goal = np.array([0.0, 10.0, 0.0])

        def point_collision(point):
            return np.linalg.norm(point - obstacles.obstacle_list[0].center) < 1.3

        def segment_collision(p1, p2):
            ts = np.linspace(0.0, 1.0, 501)
            return any(point_collision(p1 * (1.0 - t) + p2 * t) for t in ts)

        bypass = guide.find_bypass(
            start,
            goal,
            point_collision=point_collision,
            segment_collision=segment_collision,
            bounds=np.array([[-5.0, -1.0, -4.0], [5.0, 11.0, 4.0]]),
        )

        self.assertIsNotNone(bypass)
        self.assertFalse(segment_collision(start, bypass))
        self.assertFalse(segment_collision(bypass, goal))

    def test_capsule_branch_is_present_in_depth_map(self):
        obstacles = ObstacleSet()
        obstacles.add_obstacle(
            "capsule",
            radius=0.25,
            start=np.array([-1.0, 5.0, 0.0]),
            end=np.array([1.0, 5.0, 0.0]),
        )
        config = SphericalProjectionConfig(
            width=41,
            height=25,
            fov_h=np.deg2rad(100.0),
            fov_v=np.deg2rad(80.0),
            near=0.1,
            far=12.0,
            safety_radius=0.3,
        )
        guide = SphericalProjectionGuide(obstacles, config=config)

        projection = guide.render_obstacles(
            np.zeros(3), np.array([0.0, 1.0, 0.0])
        )

        center = (config.height // 2, config.width // 2)
        self.assertTrue(np.isfinite(projection.depth_map[center]))
        self.assertLess(projection.depth_map[center], 5.0)

    def test_candidate_set_preserves_left_right_up_down_sectors(self):
        config = SphericalProjectionConfig(
            width=25,
            height=17,
            fov_h=np.deg2rad(120.0),
            fov_v=np.deg2rad(100.0),
            max_candidates=8,
        )
        guide = SphericalProjectionGuide([], config=config)
        projection = guide.render_obstacles(
            np.zeros(3), np.array([0.0, 1.0, 0.0])
        )

        directions = np.asarray(guide.candidate_directions(projection))

        self.assertTrue(np.any(directions[:, 0] < 0.0))
        self.assertTrue(np.any(directions[:, 0] > 0.0))
        self.assertTrue(np.any(directions[:, 2] < 0.0))
        self.assertTrue(np.any(directions[:, 2] > 0.0))


if __name__ == "__main__":
    unittest.main()

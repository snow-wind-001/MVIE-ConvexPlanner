"""Frustum-to-spherical-map projection used as a local path guide.

The module keeps the 2-D angular map as a perception/planning front-end.  Any
waypoint proposed from the map must still pass the planner's exact 3-D point
and segment collision checks before it can be used.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable
import time

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt


@dataclass(frozen=True)
class SphericalProjectionConfig:
    width: int = 91
    height: int = 61
    fov_h: float = np.deg2rad(150.0)
    fov_v: float = np.deg2rad(110.0)
    near: float = 0.1
    far: float = 20.0
    safety_radius: float = 0.3
    max_candidates: int = 24

    def __post_init__(self):
        if self.width < 3 or self.height < 3:
            raise ValueError("projection width and height must be at least 3")
        if not 0.0 < self.fov_h < np.pi or not 0.0 < self.fov_v < np.pi:
            raise ValueError("projection FOV values must lie in (0, pi)")
        if self.near < 0.0 or self.far <= self.near:
            raise ValueError("projection range must satisfy 0 <= near < far")
        if self.safety_radius < 0.0:
            raise ValueError("safety_radius must be non-negative")


@dataclass
class SphericalProjection:
    depth_map: np.ndarray
    blocked_mask: np.ndarray
    theta_values: np.ndarray
    phi_values: np.ndarray
    origin: np.ndarray
    right: np.ndarray
    forward: np.ndarray
    up: np.ndarray


class SphericalProjectionGuide:
    """Build a spherical depth map and propose locally safe bypass waypoints."""

    def __init__(
        self,
        obstacles: Iterable,
        config: SphericalProjectionConfig | None = None,
    ):
        self.obstacles = obstacles
        self.config = config or SphericalProjectionConfig()
        self.last_projection: SphericalProjection | None = None
        self._cached_obstacle_count = -1
        self._refresh_obstacle_cache()

    def _refresh_obstacle_cache(self):
        """Group native shapes once for vectorized ray casting."""
        spheres = []
        cylinders = []
        capsules = []
        cuboids = []
        for obstacle in self.obstacles:
            shape = getattr(obstacle, "shape", "sphere")
            if shape == "sphere":
                spheres.append(obstacle)
            elif shape == "cylinder":
                cylinders.append(obstacle)
            elif shape == "capsule":
                capsules.append(obstacle)
            elif shape == "cuboid":
                cuboids.append(obstacle)
        self._sphere_centers = np.asarray(
            [obstacle.center for obstacle in spheres], dtype=float
        ).reshape(-1, 3)
        self._sphere_radii = np.asarray(
            [obstacle.radius for obstacle in spheres], dtype=float
        )
        self._cylinder_centers = np.asarray(
            [obstacle.center for obstacle in cylinders], dtype=float
        ).reshape(-1, 3)
        self._cylinder_radii = np.asarray(
            [obstacle.radius for obstacle in cylinders], dtype=float
        )
        self._cylinder_heights = np.asarray(
            [obstacle.height for obstacle in cylinders], dtype=float
        )
        self._capsule_starts = np.asarray(
            [obstacle.start for obstacle in capsules], dtype=float
        ).reshape(-1, 3)
        self._capsule_ends = np.asarray(
            [obstacle.end for obstacle in capsules], dtype=float
        ).reshape(-1, 3)
        self._capsule_radii = np.asarray(
            [obstacle.radius for obstacle in capsules], dtype=float
        )
        self._cuboids = tuple(cuboids)
        try:
            self._cached_obstacle_count = len(self.obstacles)
        except TypeError:
            self._cached_obstacle_count = (
                len(spheres) + len(cylinders) + len(capsules) + len(cuboids)
            )

    def _ensure_obstacle_cache(self):
        try:
            current_count = len(self.obstacles)
        except TypeError:
            return
        if current_count != self._cached_obstacle_count:
            self._refresh_obstacle_cache()

    @staticmethod
    def _camera_basis(forward: np.ndarray):
        forward = np.asarray(forward, dtype=float)
        norm = np.linalg.norm(forward)
        if norm < 1e-10:
            raise ValueError("forward direction must be non-zero")
        forward = forward / norm
        up_reference = np.array([0.0, 0.0, 1.0])
        if abs(np.dot(forward, up_reference)) > 0.95:
            up_reference = np.array([0.0, 1.0, 0.0])
        right = np.cross(forward, up_reference)
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        up /= np.linalg.norm(up)
        return right, forward, up

    def _angular_axes(self):
        cfg = self.config
        theta = np.linspace(-cfg.fov_h / 2.0, cfg.fov_h / 2.0, cfg.width)
        # Image row zero is the top, hence elevation decreases with row index.
        phi = np.linspace(cfg.fov_v / 2.0, -cfg.fov_v / 2.0, cfg.height)
        return theta, phi

    def _make_projection(
        self,
        depth_map: np.ndarray,
        origin: np.ndarray,
        right: np.ndarray,
        forward: np.ndarray,
        up: np.ndarray,
    ):
        theta, phi = self._angular_axes()
        projection = SphericalProjection(
            depth_map=depth_map,
            blocked_mask=self.inflate_depth_map(depth_map),
            theta_values=theta,
            phi_values=phi,
            origin=np.asarray(origin, dtype=float),
            right=right,
            forward=forward,
            up=up,
        )
        self.last_projection = projection
        return projection

    def project_points(
        self,
        points: np.ndarray,
        origin: np.ndarray,
        forward: np.ndarray,
    ) -> SphericalProjection:
        """Project world-frame point samples into a nearest-depth angular map."""
        cfg = self.config
        origin = np.asarray(origin, dtype=float)
        points = np.asarray(points, dtype=float)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError("points must have shape (N, 3)")
        right, forward, up = self._camera_basis(forward)
        relative = points - origin
        local_x = relative @ right
        local_y = relative @ forward
        local_z = relative @ up
        distance = np.linalg.norm(relative, axis=1)
        theta = np.arctan2(local_x, local_y)
        phi = np.arctan2(local_z, np.hypot(local_x, local_y))

        valid = (
            (distance >= cfg.near)
            & (distance <= cfg.far)
            & (local_y > 0.0)
            & (np.abs(theta) <= cfg.fov_h / 2.0)
            & (np.abs(phi) <= cfg.fov_v / 2.0)
        )
        u = np.rint(
            (theta[valid] + cfg.fov_h / 2.0)
            / cfg.fov_h
            * (cfg.width - 1)
        ).astype(int)
        v = np.rint(
            (cfg.fov_v / 2.0 - phi[valid])
            / cfg.fov_v
            * (cfg.height - 1)
        ).astype(int)
        depth_map = np.full((cfg.height, cfg.width), np.inf, dtype=float)
        np.minimum.at(depth_map, (v, u), distance[valid])
        return self._make_projection(
            depth_map, origin, right, forward, up
        )

    def _world_rays(self, right, forward, up):
        theta, phi = self._angular_axes()
        theta_grid, phi_grid = np.meshgrid(theta, phi)
        local_x = np.sin(theta_grid) * np.cos(phi_grid)
        local_y = np.cos(theta_grid) * np.cos(phi_grid)
        local_z = np.sin(phi_grid)
        rays = (
            local_x[..., None] * right
            + local_y[..., None] * forward
            + local_z[..., None] * up
        )
        return rays.reshape(-1, 3)

    @staticmethod
    def _select_positive_root(t1, t2, near):
        return np.where(t1 >= near, t1, np.where(t2 >= near, t2, np.inf))

    def _sphere_depth(self, origin, rays, obstacle):
        center = np.asarray(obstacle.center, dtype=float)
        radius = float(obstacle.radius)
        oc = origin - center
        linear = rays @ oc
        discriminant = linear * linear - (np.dot(oc, oc) - radius * radius)
        valid = discriminant >= 0.0
        root = np.sqrt(np.maximum(discriminant, 0.0))
        near_t = -linear - root
        far_t = -linear + root
        depth = self._select_positive_root(near_t, far_t, self.config.near)
        return np.where(valid, depth, np.inf)

    def _sphere_depth_batch(self, origin, rays, centers, radii):
        if len(centers) == 0:
            return np.full(len(rays), np.inf)
        nearby = np.linalg.norm(centers - origin[None, :], axis=1) - radii <= self.config.far
        if not np.any(nearby):
            return np.full(len(rays), np.inf)
        centers = centers[nearby]
        radii = radii[nearby]
        offsets = origin[None, :] - centers
        linear = rays @ offsets.T
        constant = np.einsum("ij,ij->i", offsets, offsets) - radii * radii
        discriminant = linear * linear - constant[None, :]
        valid = discriminant >= 0.0
        root = np.sqrt(np.maximum(discriminant, 0.0))
        near_t = -linear - root
        far_t = -linear + root
        candidate = np.where(
            near_t >= self.config.near,
            near_t,
            np.where(far_t >= self.config.near, far_t, np.inf),
        )
        candidate = np.where(valid, candidate, np.inf)
        return np.min(candidate, axis=1)

    def _cuboid_depth(self, origin, rays, obstacle):
        center = np.asarray(obstacle.center, dtype=float)
        half = np.asarray(obstacle.size, dtype=float) / 2.0
        lower = center - half
        upper = center + half
        t_min = np.full(len(rays), -np.inf)
        t_max = np.full(len(rays), np.inf)
        valid = np.ones(len(rays), dtype=bool)
        for axis in range(3):
            direction = rays[:, axis]
            parallel = np.abs(direction) < 1e-12
            valid &= ~(parallel & ((origin[axis] < lower[axis]) | (origin[axis] > upper[axis])))
            nonparallel = ~parallel
            entry = np.full(len(rays), -np.inf)
            exit_ = np.full(len(rays), np.inf)
            values1 = (lower[axis] - origin[axis]) / direction[nonparallel]
            values2 = (upper[axis] - origin[axis]) / direction[nonparallel]
            entry[nonparallel] = np.minimum(values1, values2)
            exit_[nonparallel] = np.maximum(values1, values2)
            t_min = np.maximum(t_min, entry)
            t_max = np.minimum(t_max, exit_)
        valid &= t_max >= np.maximum(t_min, self.config.near)
        depth = np.where(t_min >= self.config.near, t_min, t_max)
        return np.where(valid & (depth >= self.config.near), depth, np.inf)

    def _cylinder_depth(self, origin, rays, obstacle):
        center = np.asarray(obstacle.center, dtype=float)
        radius = float(obstacle.radius)
        half_height = float(obstacle.height) / 2.0
        local_origin = origin - center
        depth = np.full(len(rays), np.inf)

        a = rays[:, 0] ** 2 + rays[:, 1] ** 2
        linear = local_origin[0] * rays[:, 0] + local_origin[1] * rays[:, 1]
        constant = local_origin[0] ** 2 + local_origin[1] ** 2 - radius ** 2
        discriminant = linear * linear - a * constant
        side_valid = (a > 1e-12) & (discriminant >= 0.0)
        root = np.sqrt(np.maximum(discriminant, 0.0))
        for candidate in (
            (-linear - root) / np.where(a > 1e-12, a, 1.0),
            (-linear + root) / np.where(a > 1e-12, a, 1.0),
        ):
            z = local_origin[2] + candidate * rays[:, 2]
            valid = side_valid & (candidate >= self.config.near) & (np.abs(z) <= half_height)
            depth = np.minimum(depth, np.where(valid, candidate, np.inf))

        nonparallel_caps = np.abs(rays[:, 2]) > 1e-12
        for cap_z in (-half_height, half_height):
            candidate = (cap_z - local_origin[2]) / np.where(
                nonparallel_caps, rays[:, 2], 1.0
            )
            x = local_origin[0] + candidate * rays[:, 0]
            y = local_origin[1] + candidate * rays[:, 1]
            valid = (
                nonparallel_caps
                & (candidate >= self.config.near)
                & (x * x + y * y <= radius * radius)
            )
            depth = np.minimum(depth, np.where(valid, candidate, np.inf))
        return depth

    def _cylinder_depth_batch(self, origin, rays):
        if len(self._cylinder_centers) == 0:
            return np.full(len(rays), np.inf)
        half_heights_all = self._cylinder_heights / 2.0
        bounds = np.hypot(self._cylinder_radii, half_heights_all)
        nearby = (
            np.linalg.norm(self._cylinder_centers - origin[None, :], axis=1)
            - bounds
            <= self.config.far
        )
        if not np.any(nearby):
            return np.full(len(rays), np.inf)
        local = origin[None, :] - self._cylinder_centers[nearby]
        radii = self._cylinder_radii[nearby]
        half_heights = half_heights_all[nearby]
        depth = np.full((len(rays), len(local)), np.inf)

        quadratic = rays[:, 0] ** 2 + rays[:, 1] ** 2
        linear = (
            rays[:, 0, None] * local[None, :, 0]
            + rays[:, 1, None] * local[None, :, 1]
        )
        constant = (
            local[:, 0] ** 2 + local[:, 1] ** 2 - radii * radii
        )
        discriminant = linear * linear - quadratic[:, None] * constant[None, :]
        side_valid = (quadratic[:, None] > 1e-12) & (discriminant >= 0.0)
        root = np.sqrt(np.maximum(discriminant, 0.0))
        denominator = np.where(quadratic > 1e-12, quadratic, 1.0)[:, None]
        for candidate in ((-linear - root) / denominator, (-linear + root) / denominator):
            z = local[None, :, 2] + candidate * rays[:, 2, None]
            valid = (
                side_valid
                & (candidate >= self.config.near)
                & (np.abs(z) <= half_heights[None, :])
            )
            depth = np.minimum(depth, np.where(valid, candidate, np.inf))

        cap_rays = np.abs(rays[:, 2]) > 1e-12
        cap_denominator = np.where(cap_rays, rays[:, 2], 1.0)[:, None]
        for sign in (-1.0, 1.0):
            candidate = (
                sign * half_heights[None, :] - local[None, :, 2]
            ) / cap_denominator
            x = local[None, :, 0] + candidate * rays[:, 0, None]
            y = local[None, :, 1] + candidate * rays[:, 1, None]
            valid = (
                cap_rays[:, None]
                & (candidate >= self.config.near)
                & (x * x + y * y <= radii[None, :] ** 2)
            )
            depth = np.minimum(depth, np.where(valid, candidate, np.inf))
        return np.min(depth, axis=1)

    def _capsule_depth(self, origin, rays, obstacle):
        """Intersect rays with an arbitrarily oriented finite capsule."""
        start = np.asarray(obstacle.start, dtype=float)
        end = np.asarray(obstacle.end, dtype=float)
        radius = float(obstacle.radius)
        axis = end - start
        axis_norm_sq = float(np.dot(axis, axis))
        if axis_norm_sq <= 1e-20:
            return self._sphere_depth(
                origin,
                rays,
                type("SphereProxy", (), {"center": start, "radius": radius})(),
            )

        relative = origin - start
        ray_axis_dot = rays @ axis
        axis_relative_dot = float(np.dot(axis, relative))
        ray_relative_dot = rays @ relative
        relative_norm_sq = float(np.dot(relative, relative))
        quadratic = axis_norm_sq - ray_axis_dot * ray_axis_dot
        linear = (
            axis_norm_sq * ray_relative_dot
            - axis_relative_dot * ray_axis_dot
        )
        constant = (
            axis_norm_sq * relative_norm_sq
            - axis_relative_dot * axis_relative_dot
            - radius * radius * axis_norm_sq
        )
        discriminant = linear * linear - quadratic * constant
        side_valid = (quadratic > 1e-12) & (discriminant >= 0.0)
        root = np.sqrt(np.maximum(discriminant, 0.0))
        depth = np.full(len(rays), np.inf)
        for candidate in (
            (-linear - root) / np.where(quadratic > 1e-12, quadratic, 1.0),
            (-linear + root) / np.where(quadratic > 1e-12, quadratic, 1.0),
        ):
            axial = axis_relative_dot + candidate * ray_axis_dot
            valid = (
                side_valid
                & (candidate >= self.config.near)
                & (axial >= 0.0)
                & (axial <= axis_norm_sq)
            )
            depth = np.minimum(depth, np.where(valid, candidate, np.inf))

        # A capsule has spherical end caps.  Include both roots so an origin
        # inside a cap still obtains the positive exit depth.
        for center in (start, end):
            offset = origin - center
            cap_linear = rays @ offset
            cap_discriminant = (
                cap_linear * cap_linear
                - (np.dot(offset, offset) - radius * radius)
            )
            cap_valid = cap_discriminant >= 0.0
            cap_root = np.sqrt(np.maximum(cap_discriminant, 0.0))
            cap_depth = self._select_positive_root(
                -cap_linear - cap_root,
                -cap_linear + cap_root,
                self.config.near,
            )
            depth = np.minimum(
                depth, np.where(cap_valid, cap_depth, np.inf)
            )
        return depth

    def _capsule_depth_batch(self, origin, rays):
        if len(self._capsule_starts) == 0:
            return np.full(len(rays), np.inf)
        midpoints = (self._capsule_starts + self._capsule_ends) / 2.0
        bounds = (
            np.linalg.norm(self._capsule_ends - self._capsule_starts, axis=1) / 2.0
            + self._capsule_radii
        )
        nearby = (
            np.linalg.norm(midpoints - origin[None, :], axis=1) - bounds
            <= self.config.far
        )
        if not np.any(nearby):
            return np.full(len(rays), np.inf)
        starts = self._capsule_starts[nearby]
        ends = self._capsule_ends[nearby]
        radii = self._capsule_radii[nearby]
        axes = ends - starts
        axis_norm_sq = np.einsum("ij,ij->i", axes, axes)
        relative = origin[None, :] - starts
        ray_axis_dot = rays @ axes.T
        axis_relative_dot = np.einsum("ij,ij->i", axes, relative)
        ray_relative_dot = rays @ relative.T
        relative_norm_sq = np.einsum("ij,ij->i", relative, relative)
        quadratic = axis_norm_sq[None, :] - ray_axis_dot * ray_axis_dot
        linear = (
            axis_norm_sq[None, :] * ray_relative_dot
            - axis_relative_dot[None, :] * ray_axis_dot
        )
        constant = (
            axis_norm_sq * relative_norm_sq
            - axis_relative_dot * axis_relative_dot
            - radii * radii * axis_norm_sq
        )
        discriminant = linear * linear - quadratic * constant[None, :]
        side_valid = (quadratic > 1e-12) & (discriminant >= 0.0)
        root = np.sqrt(np.maximum(discriminant, 0.0))
        depth = np.full_like(quadratic, np.inf)
        denominator = np.where(quadratic > 1e-12, quadratic, 1.0)
        for candidate in ((-linear - root) / denominator, (-linear + root) / denominator):
            axial = axis_relative_dot[None, :] + candidate * ray_axis_dot
            valid = (
                side_valid
                & (candidate >= self.config.near)
                & (axial >= 0.0)
                & (axial <= axis_norm_sq[None, :])
            )
            depth = np.minimum(depth, np.where(valid, candidate, np.inf))

        cap_centers = np.vstack((starts, ends))
        cap_radii = np.concatenate((radii, radii))
        cap_depth = self._sphere_depth_batch(
            origin, rays, cap_centers, cap_radii
        )
        return np.minimum(np.min(depth, axis=1), cap_depth)

    def render_obstacles(
        self,
        origin: np.ndarray,
        forward: np.ndarray,
        deadline: float | None = None,
    ) -> SphericalProjection | None:
        """Ray-cast the simulator's analytic obstacles into an angular map."""
        cfg = self.config
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        self._ensure_obstacle_cache()
        origin = np.asarray(origin, dtype=float)
        right, forward, up = self._camera_basis(forward)
        rays = self._world_rays(right, forward, up)
        depth = np.full(len(rays), np.inf)
        native_batches = (
            self._sphere_depth_batch(
                origin, rays, self._sphere_centers, self._sphere_radii
            ),
            self._cylinder_depth_batch(origin, rays),
            self._capsule_depth_batch(origin, rays),
        )
        for candidate in native_batches:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            depth = np.minimum(depth, candidate)
        # Cuboids are uncommon in the forest workload and remain in the
        # compact slab loop to avoid allocating a third-order ray/box tensor.
        for obstacle in self._cuboids:
            if deadline is not None and time.perf_counter() >= deadline:
                return None
            depth = np.minimum(depth, self._cuboid_depth(origin, rays, obstacle))
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        depth[(depth < cfg.near) | (depth > cfg.far)] = np.inf
        return self._make_projection(
            depth.reshape(cfg.height, cfg.width), origin, right, forward, up
        )

    @staticmethod
    def _ellipse_structure(radius_v: int, radius_u: int):
        yy, xx = np.ogrid[-radius_v : radius_v + 1, -radius_u : radius_u + 1]
        return (yy / max(radius_v, 1)) ** 2 + (xx / max(radius_u, 1)) ** 2 <= 1.0

    def inflate_depth_map(self, depth_map: np.ndarray) -> np.ndarray:
        """Inflate obstacles using the range-dependent angular footprint."""
        cfg = self.config
        depth_map = np.asarray(depth_map, dtype=float)
        if depth_map.shape != (cfg.height, cfg.width):
            raise ValueError("depth_map shape does not match projection config")
        occupied = np.isfinite(depth_map)
        if not np.any(occupied):
            return np.zeros_like(occupied)
        if cfg.safety_radius == 0.0:
            return occupied

        delta_theta = cfg.fov_h / (cfg.width - 1)
        delta_phi = cfg.fov_v / (cfg.height - 1)
        angles = np.zeros_like(depth_map)
        angles[occupied] = np.arcsin(
            np.minimum(0.999, cfg.safety_radius / np.maximum(depth_map[occupied], 1e-9))
        )
        radius_u = np.maximum(1, np.ceil(angles / delta_theta).astype(int))
        radius_v = np.maximum(1, np.ceil(angles / delta_phi).astype(int))
        blocked = occupied.copy()
        radius_pairs = np.unique(
            np.column_stack((radius_v[occupied], radius_u[occupied])), axis=0
        )
        for rv, ru in radius_pairs:
            source = occupied & (radius_v == rv) & (radius_u == ru)
            blocked |= binary_dilation(
                source, structure=self._ellipse_structure(int(rv), int(ru))
            )
        return blocked

    def build_navigability_graph(self, projection: SphericalProjection):
        """Build the row-interval graph proposed by the projection scheme."""
        free = ~projection.blocked_mask
        nodes = []
        rows = []
        node_id = 0
        for v, row in enumerate(free):
            row_nodes = []
            padded = np.r_[False, row, False].astype(int)
            changes = np.diff(padded)
            starts = np.flatnonzero(changes == 1)
            ends = np.flatnonzero(changes == -1) - 1
            for start, end in zip(starts, ends):
                if end - start + 1 < 2:
                    continue
                nodes.append(
                    {
                        "id": node_id,
                        "row": v,
                        "u_start": int(start),
                        "u_end": int(end),
                        "u_center": float((start + end) / 2.0),
                        "width": int(end - start + 1),
                    }
                )
                row_nodes.append(node_id)
                node_id += 1
            rows.append(row_nodes)
        by_id = {node["id"]: node for node in nodes}
        edges = []
        for previous, current in zip(rows[:-1], rows[1:]):
            for left in previous:
                for right in current:
                    a, b = by_id[left], by_id[right]
                    overlap = min(a["u_end"], b["u_end"]) - max(
                        a["u_start"], b["u_start"]
                    )
                    if overlap >= 0:
                        edges.append((left, right))
        return {"nodes": nodes, "edges": edges}

    def candidate_directions(self, projection: SphericalProjection):
        free = ~projection.blocked_mask
        if not np.any(free):
            return []
        clearance = distance_transform_edt(free)
        theta_grid, phi_grid = np.meshgrid(
            projection.theta_values, projection.phi_values
        )
        angular_error = theta_grid ** 2 + phi_grid ** 2
        edge_v = np.minimum(
            np.arange(self.config.height),
            np.arange(self.config.height)[::-1],
        )[:, None]
        edge_u = np.minimum(
            np.arange(self.config.width),
            np.arange(self.config.width)[::-1],
        )[None, :]
        edge_clearance = np.minimum(edge_v, edge_u)
        # Prefer small steering angles while giving a modest bonus to the
        # center of a wide free channel.  A strong clearance term pushes the
        # candidate too far from the goal and slows exact verification.
        score = angular_error - 0.005 * clearance
        score += np.where(edge_clearance < 2, 1.0, 0.0)
        score[~free] = np.inf

        selected = []
        selected_pixels = []

        def add_pixel(flat_index):
            v, u = np.unravel_index(flat_index, score.shape)
            if any((v - pv) ** 2 + (u - pu) ** 2 < 9 for pv, pu in selected_pixels):
                return False
            theta = projection.theta_values[u]
            phi = projection.phi_values[v]
            local = np.array(
                [
                    np.sin(theta) * np.cos(phi),
                    np.cos(theta) * np.cos(phi),
                    np.sin(phi),
                ]
            )
            world = (
                local[0] * projection.right
                + local[1] * projection.forward
                + local[2] * projection.up
            )
            selected.append(world / np.linalg.norm(world))
            selected_pixels.append((v, u))
            return True

        def add_best(mask):
            masked_score = np.where(mask, score, np.inf)
            for flat_index in np.argsort(masked_score, axis=None):
                if not np.isfinite(masked_score.flat[flat_index]):
                    return
                if add_pixel(flat_index):
                    return

        # Preserve genuine 3-D maneuverability even on a coarse realtime map.
        # The four cardinal sectors are considered before score-only filling,
        # so left/right/up/down do not get crowded out by several nearly
        # identical low-angle pixels on one side of an obstacle.
        cardinal_masks = (
            (theta_grid < 0.0) & (np.abs(theta_grid) >= np.abs(phi_grid)),  # left
            (theta_grid > 0.0) & (np.abs(theta_grid) >= np.abs(phi_grid)),  # right
            (phi_grid > 0.0) & (np.abs(phi_grid) > np.abs(theta_grid)),     # up
            (phi_grid < 0.0) & (np.abs(phi_grid) > np.abs(theta_grid)),     # down
        )
        sector_order = sorted(
            cardinal_masks,
            key=lambda mask: float(np.min(np.where(mask & free, score, np.inf))),
        )
        for mask in sector_order:
            if len(selected) >= self.config.max_candidates:
                break
            add_best(mask & free)

        if len(selected) < self.config.max_candidates:
            add_best(free)

        order = np.argsort(score, axis=None)
        for flat_index in order:
            if len(selected) >= self.config.max_candidates:
                break
            if not np.isfinite(score.flat[flat_index]):
                break
            add_pixel(flat_index)
        return selected

    def find_bypass(
        self,
        p1: np.ndarray,
        p2: np.ndarray,
        point_collision: Callable[[np.ndarray], bool],
        segment_collision: Callable[[np.ndarray, np.ndarray], bool],
        bounds: np.ndarray | None = None,
        distance_fractions: tuple[float, ...] = (0.35, 0.5, 0.65, 0.8),
        max_verifications: int | None = None,
        deadline: float | None = None,
    ) -> np.ndarray | None:
        """Return the first projection-guided waypoint verified safe in 3-D."""
        p1 = np.asarray(p1, dtype=float)
        p2 = np.asarray(p2, dtype=float)
        segment = p2 - p1
        segment_length = np.linalg.norm(segment)
        if segment_length < 1e-8:
            return None
        projection = self.render_obstacles(p1, segment, deadline=deadline)
        if projection is None:
            return None
        directions = self.candidate_directions(projection)
        if deadline is not None and time.perf_counter() >= deadline:
            return None
        distances = np.unique(
            np.clip(
                segment_length * np.asarray(distance_fractions, dtype=float),
                max(self.config.near * 2.0, 0.5),
                min(self.config.far, max(segment_length * 0.8, 0.5)),
            )
        )
        if bounds is not None:
            bounds = np.asarray(bounds, dtype=float)
        verification_count = 0
        for direction in directions:
            for distance in distances:
                if deadline is not None and time.perf_counter() >= deadline:
                    return None
                if (
                    max_verifications is not None
                    and verification_count >= max_verifications
                ):
                    return None
                verification_count += 1
                candidate = p1 + float(distance) * direction
                if bounds is not None:
                    candidate = np.clip(candidate, bounds[0] + 0.05, bounds[1] - 0.05)
                if np.linalg.norm(candidate - p1) < 0.2 or np.linalg.norm(candidate - p2) < 0.2:
                    continue
                if point_collision(candidate):
                    continue
                if deadline is not None and time.perf_counter() >= deadline:
                    return None
                if segment_collision(p1, candidate):
                    continue
                if deadline is not None and time.perf_counter() >= deadline:
                    return None
                if not segment_collision(candidate, p2):
                    if deadline is not None and time.perf_counter() >= deadline:
                        return None
                    return candidate
        return None

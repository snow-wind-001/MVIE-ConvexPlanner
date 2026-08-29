import unittest

import numpy as np

from firi.geometry import ConvexPolytope
from firi.planning.mvie import MVIE_SOCP


def axis_aligned_box(center, half_extent=1.0):
    halfspaces = []
    for axis in range(3):
        normal = np.zeros(3)
        normal[axis] = 1.0
        halfspaces.append(
            np.r_[normal, -(center[axis] + half_extent)]
        )
        halfspaces.append(
            np.r_[-normal, center[axis] - half_extent]
        )
    return ConvexPolytope(halfspaces=np.asarray(halfspaces))


class MVIERegressionTests(unittest.TestCase):
    def test_mvie_is_translation_invariant_for_identical_boxes(self):
        solver = MVIE_SOCP(3)
        origin_ellipsoid = solver.compute(axis_aligned_box(np.zeros(3)))
        shift = np.array([10.0, 20.0, 30.0])
        shifted_ellipsoid = solver.compute(axis_aligned_box(shift))

        self.assertTrue(
            np.allclose(origin_ellipsoid.center, np.zeros(3), atol=1e-5)
        )
        self.assertTrue(np.allclose(shifted_ellipsoid.center, shift, atol=1e-5))
        self.assertTrue(
            np.allclose(
                origin_ellipsoid.Q,
                shifted_ellipsoid.Q,
                rtol=1e-4,
                atol=1e-5,
            )
        )
        self.assertTrue(
            np.isclose(
                origin_ellipsoid.volume(),
                shifted_ellipsoid.volume(),
                rtol=1e-4,
            )
        )
        self.assertTrue(
            np.allclose(np.diag(origin_ellipsoid.Q), np.ones(3), atol=1e-4)
        )


if __name__ == "__main__":
    unittest.main()

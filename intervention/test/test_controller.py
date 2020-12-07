import math
import unittest

import numpy as np

from .. import controller


class Test(unittest.TestCase):
    def test_turning_radius(self):
        self.assertEqual(controller._turning_radius_to(1.0, 0.0), 0.5)
        self.assertEqual(controller._turning_radius_to(-1.0, 0.0), 0.5)
        self.assertEqual(controller._turning_radius_to(1.0, 1.0), 1.0)
        self.assertEqual(controller._turning_radius_to(-1.0, 1.0), 1.0)
        self.assertEqual(controller._turning_radius_to(0, 1.0), math.inf)

    def test_waypoint_interpolation(self):
        waypoints_distances_and_targets = [
            ([[0.5, 0]], 1.0, [1.0, 0]),
            ([[-0.5, 0]], 1.0, [-1.0, 0]),
            ([[0.5, 0], [1.5, 0]], 1.0, [1.0, 0]),
            ([[-0.5, 0], [10.0, 0]], 1.0, [0.0, 0]),
            ([[0.5, 0], [0.5, 1.0]], 1.0, [0.5, 0.5]),
            ([[0.3, -0.3]], 20.0, [math.sqrt(20 ** 2 / 2), -math.sqrt(20 ** 2 / 2)]),
        ]

        for waypoints, distance, target in waypoints_distances_and_targets:
            tx, ty = target
            x, y = controller._interpolate_waypoint_n_meters_ahead(
                np.array(waypoints), distance
            )
            self.assertAlmostEqual(tx, x, places=5)
            self.assertAlmostEqual(ty, y, places=5)

    def test_waypoint_lookahead(self):
        waypoints_distances_and_targets = [
            ([[0.5, 0]], 1.0, [1.0, 0]),
            ([[-0.5, 0]], 1.0, [-1.0, 0]),
            ([[0.5, 0], [1.5, 0]], 1.0, [1.0, 0]),
            ([[-0.5, 0], [10.0, 0]], 1.0, [1.0, 0]),
            ([[0.5, 0], [0.5, 0.5]], 1.0, [0.5, math.sqrt(1 ** 2 - 0.5 ** 2)]),
            ([[0.5, 0], [0.5, 0.3], [-1.5, -0.1]], 1.0, [-1.0, 0.0]),
            ([[-0.2, 0.5], [-0.2, 1.0]], 1.0, [-0.2, math.sqrt(1 ** 2 - 0.2 ** 2)]),
            ([[0.2, 0.5], [0.2, 1.0]], 1.0, [0.2, math.sqrt(1 ** 2 - 0.2 ** 2)]),
            ([[0.2, -0.5], [0.2, -1.0]], 1.0, [0.2, -math.sqrt(1 ** 2 - 0.2 ** 2)]),
            (
                [[-20, 20], [20, -20], [500, 500]],
                10.0,
                [-math.sqrt(10 ** 2 / 2), math.sqrt(10 ** 2 / 2)],
            ),
            (
                [[-5, 5], [20, -20], [500, 500]],
                10.0,
                [math.sqrt(10 ** 2 / 2), -math.sqrt(10 ** 2 / 2)],
            ),
        ]

        for waypoints, distance, target in waypoints_distances_and_targets:
            tx, ty = target
            x, y = controller._lookahead_trajectory_n_meters_ahead(
                np.array(waypoints), distance
            )
            self.assertAlmostEqual(tx, x, places=5)
            self.assertAlmostEqual(ty, y, places=5)

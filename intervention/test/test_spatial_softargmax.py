from typing import Tuple

import unittest
import torch

from ..models.spatial_softargmax import SpatialSoftargmax


class Test(unittest.TestCase):
    EPSILON = 1e-5

    def test_singular_argmax_location(self):
        ssam = SpatialSoftargmax(21, 7, 1)

        for (x, y) in [(0, 0), (6, 19), (1, 0), (0, 1), (4, 12)]:
            t = torch.zeros(1, 1, 21, 7)
            t[..., y, x] = 1000

            coord, _ = ssam(t)
            argmax_x, argmax_y = ssam.to_width_height_range(coord)[0, 0, :].tolist()

            self.assertTrue(abs(x - argmax_x) < Test.EPSILON)
            self.assertTrue(abs(y - argmax_y) < Test.EPSILON)

    def test_two_argmax_location(self):
        ssam = SpatialSoftargmax(21, 7, 1)

        for ((x1, y1), (x2, y2), (x_target, y_target)) in [
            ((0, 0), (6, 20), (3, 10)),
            ((0, 0), (6, 0), (3, 0)),
            ((2, 12), (3, 20), (2.5, 16)),
        ]:
            t = torch.zeros(1, 1, 21, 7)
            t[..., y1, x1] = 1000
            t[..., y2, x2] = 1000

            coord, _ = ssam(t)
            argmax_x, argmax_y = ssam.to_width_height_range(coord)[0, 0, :].tolist()

            self.assertTrue(abs(x_target - argmax_x) < Test.EPSILON)
            self.assertTrue(abs(y_target - argmax_y) < Test.EPSILON)

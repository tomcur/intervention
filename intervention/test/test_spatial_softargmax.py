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

    def test_heatmap(self):
        ssam = SpatialSoftargmax(21, 7, 1)

        for ((x1, y1), (x2, y2), (x_target, y_target)) in [
            ((0, 0), (6, 20), (3, 10)),
            ((0, 0), (6, 0), (3, 0)),
            ((2, 12), (3, 20), (2.5, 16)),
        ]:
            t = torch.zeros(1, 1, 21, 7)
            t[..., y1, x1] = 1000
            t[..., y2, x2] = 1000

            _, heatmap = ssam(t)

            self.assertTrue(abs(1.0 - heatmap.sum().item()) < Test.EPSILON)

            self.assertTrue(abs(0.5 - heatmap[0, 0, y1, x1].item()) < Test.EPSILON)
            self.assertTrue(abs(0.5 - heatmap[0, 0, y2, x2].item()) < Test.EPSILON)


class TestGradients(unittest.TestCase):
    def test_argmax_gradient_direction(self):
        tensor = torch.tensor(
            [
                [
                    [
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0],
                        [1.0, 1.0, 1.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0],
                    ]
                ]
            ],
            requires_grad=True,
        )
        ssam = SpatialSoftargmax(7, 5, 1)
        coords, _ = ssam(tensor)

        # Set target expected value to positive-X.
        diff = torch.abs(coords - torch.tensor([1.0, 0.0]))
        loss = torch.abs(diff.pow(2)).mean()
        loss.backward()

        # We expect the negative gradient to be to positive-X.
        expected_positive_loss = torch.tensor(
            [
                [
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                    [True, True, False, False, False],
                ]
            ]
        )

        self.assertTrue(torch.all(expected_positive_loss.eq(tensor.grad > 0)))

        # Make a small step in direction of negative loss.
        with torch.no_grad():
            tensor -= 0.1 * tensor.grad
            grad = tensor.grad.detach().clone()
            tensor.grad.zero_()

        new_coords, _ = ssam(tensor)
        old_argmax_x, old_argmax_y = coords[0, 0].tolist()
        new_argmax_x, new_argmax_y = new_coords[0, 0].tolist()

        # We expect the argmax X to have moved in direction of positive-X...
        self.assertGreater(new_argmax_x, old_argmax_x)

        # ... but for the argmax Y to stay the same.
        self.assertAlmostEqual(old_argmax_y, new_argmax_y, places=6)

        # If instead we make a step in direction of positive loss...
        with torch.no_grad():
            tensor += 0.2 * grad

        new_coords, _ = ssam(tensor)
        new_argmax_x, new_argmax_y = new_coords[0, 0].tolist()

        # We expect the argmax X to have moved to negative-X...
        self.assertLess(new_argmax_x, old_argmax_x)

        # ... and for the argmax Y to still stay the same.
        self.assertAlmostEqual(old_argmax_y, new_argmax_y, places=6)

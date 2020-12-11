import unittest

import torch

from ..train import train


class Test(unittest.TestCase):
    def test_four_hot(self):
        test = [
            (1.0, -0.2, 2, 2),
            (1.0, 1.0, 2, 2),
            (1.0, 0.0, 2, 2),
            (1.0, -1.0, 2, 2),
            (-1.0, -1.0, 2, 2),
            (0, 0, 2, 2),
            (0.5, -0.2, 100, 50),
            (0.5, -0.2, 96, 40),
            (0.5, -0.2, 33, 11),
        ]

        for (target_x, target_y, width, height) in test:
            heatmap = train.cross_entropy_four_hot(target_x, target_y, width, height)
            self.assertAlmostEqual(float(torch.sum(heatmap)), 1.0)

            ev_x = 0
            ev_y = 0
            for (y, row) in enumerate(heatmap):
                for (x, p) in enumerate(row):
                    ev_x += p * (x * 2.0 / (width - 1) - 1.0)
                    ev_y += p * (y * 2.0 / (height - 1) - 1.0)

            self.assertAlmostEqual(target_x, ev_x.item())
            self.assertAlmostEqual(target_y, ev_y.item())

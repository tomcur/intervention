import unittest

import numpy as np

from .. import physics


class Test(unittest.TestCase):
    def test_cot(self):
        for x in np.arange(0.1, 1.5, 0.1):
            self.assertAlmostEqual(x, physics._arccot(physics._cot(x)), places=5)
            self.assertAlmostEqual(x, physics._cot(physics._arccot(x)), places=5)

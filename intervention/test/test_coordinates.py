import unittest

from .. import coordinates


class Test(unittest.TestCase):
    EPSILON = 1e-10

    def test_image_roundtrip(self):
        im_x, im_y = coordinates.world_coordinate_to_image_coordinate(
            20, -10, 0, 0, 0, -1
        )

        ego_x, ego_z = coordinates.image_coordinate_to_ego_coordinate(im_x, im_y)

        self.assertTrue(abs(ego_x - 20.0) < Test.EPSILON)
        self.assertTrue(abs(ego_z - 10.0) < Test.EPSILON)

import unittest
import numpy as np

from .. import coordinates


class TestTransform(unittest.TestCase):
    def test_carla_to_opencv_singular(self):
        carla_coord = np.array([1, 2, 3])
        opencv_coord = np.array([1, -3, 2])
        self.assertTrue(
            (coordinates.carla_to_opencv(carla_coord) == opencv_coord).all()
        )

    def test_carla_to_opencv_multiple(self):
        carla_coord = np.array([[1, 2, 3], [4, 5, 6]])
        opencv_coord = np.array([[1, -3, 2], [4, -6, 5]])
        self.assertTrue(
            (coordinates.carla_to_opencv(carla_coord) == opencv_coord).all()
        )


class Test(unittest.TestCase):
    EPSILON = 1e-10

    def test_image_roundtrip(self):
        im_x, im_y = coordinates.world_coordinate_to_image_coordinate(
            20, -10, 0, 0, 0, -1
        )

        ego_x, ego_z = coordinates.image_coordinate_to_ego_coordinate(im_x, im_y)

        self.assertTrue(abs(ego_x - 20.0) < Test.EPSILON)
        self.assertTrue(abs(ego_z - 10.0) < Test.EPSILON)

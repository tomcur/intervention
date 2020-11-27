import math
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

    def test_world_to_ego_transform_1(self):
        ego_x, ego_y = coordinates.world_coordinate_to_ego_coordinate(
            20, 10, 0, 0, 0, 1
        )

        self.assertTrue(abs(-20 - ego_x) < Test.EPSILON)
        self.assertTrue(abs(10 - ego_y) < Test.EPSILON)

    def test_world_to_ego_transform_2(self):
        ego_x, ego_y = coordinates.world_coordinate_to_ego_coordinate(
            20, 10, 0, 0, 0, -1
        )

        self.assertTrue(abs(20 - ego_x) < Test.EPSILON)
        self.assertTrue(abs(-10 - ego_y) < Test.EPSILON)

    def test_world_to_ego_transform_3(self):
        ego_x, ego_y = coordinates.world_coordinate_to_ego_coordinate(
            20, -10, 5, 2, 0, -1
        )

        self.assertTrue(abs(15 - ego_x) < Test.EPSILON)
        self.assertTrue(abs(12 - ego_y) < Test.EPSILON)

    def test_world_to_ego_transform_4(self):
        ego_x, ego_y = coordinates.world_coordinate_to_ego_coordinate(
            20, 10, 0, 0, 1, 0
        )

        self.assertTrue(abs(10 - ego_x) < Test.EPSILON)
        self.assertTrue(abs(20 - ego_y) < Test.EPSILON)

    def test_world_to_ego_transform_5(self):
        ego_x, ego_y = coordinates.world_coordinate_to_ego_coordinate(
            20, 10, 0, 0, -1, 0
        )

        self.assertTrue(abs(-10 - ego_x) < Test.EPSILON)
        self.assertTrue(abs(-20 - ego_y) < Test.EPSILON)

    def test_ego_transform_roundtrip_1(self):
        ego_x, ego_y = coordinates.world_coordinate_to_ego_coordinate(
            20, -10, 0, 0, 0, -1
        )

        world_x, world_y = coordinates.ego_coordinate_to_world_coordinate(
            ego_x, ego_y, 0, 0, 0, -1
        )

        self.assertTrue(abs(world_x - 20) < Test.EPSILON)
        self.assertTrue(abs(world_y - -10) < Test.EPSILON)

    def test_ego_transform_roundtrip_2(self):
        ego_x, ego_y = coordinates.world_coordinate_to_ego_coordinate(
            20, -10, 5, 10, math.sqrt(1 / 2), -math.sqrt(1 / 2)
        )

        world_x, world_y = coordinates.ego_coordinate_to_world_coordinate(
            ego_x, ego_y, 5, 10, math.sqrt(1 / 2), -math.sqrt(1 / 2)
        )

        self.assertTrue(abs(world_x - 20) < Test.EPSILON)
        self.assertTrue(abs(world_y - -10) < Test.EPSILON)

    def test_image_transform_roundtrip_1(self):
        ego_x, ego_y = coordinates.world_coordinate_to_ego_coordinate(
            20, -10, 0, 0, 0, -1
        )

        im_x, im_y = coordinates.ego_coordinate_to_image_coordinate(ego_x, ego_y,)

        ego_x2, ego_y2 = coordinates.image_coordinate_to_ego_coordinate(im_x, im_y)

        im_x2, im_y2 = coordinates.ego_coordinate_to_image_coordinate(ego_x2, ego_y2,)

        self.assertTrue(abs(ego_x - ego_x2) < Test.EPSILON)
        self.assertTrue(abs(ego_y - ego_y2) < Test.EPSILON)

        self.assertTrue(abs(im_x - im_x2) < Test.EPSILON)
        self.assertTrue(abs(im_y - im_y2) < Test.EPSILON)

    def test_image_transform_roundtrip_2(self):
        ego_x, ego_y = coordinates.world_coordinate_to_ego_coordinate(
            20, -10, 5, 10, 0, -1,
        )

        im_x, im_y = coordinates.ego_coordinate_to_image_coordinate(
            ego_x, ego_y, forward_offset=6.0
        )

        ego_x2, ego_y2 = coordinates.image_coordinate_to_ego_coordinate(
            im_x, im_y, forward_offset=6.0
        )

        im_x2, im_y2 = coordinates.ego_coordinate_to_image_coordinate(
            ego_x2, ego_y2, forward_offset=6.0
        )

        self.assertTrue(abs(ego_x - ego_x2) < Test.EPSILON)
        self.assertTrue(abs(ego_y - ego_y2) < Test.EPSILON)

        self.assertTrue(abs(im_x - im_x2) < Test.EPSILON)
        self.assertTrue(abs(im_y - im_y2) < Test.EPSILON)

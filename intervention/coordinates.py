from typing import Tuple

import numpy as np
import cv2

CAMERA_Z_OFFSET = 1.4
CAMERA_FORWARD_OFFSET = 2.0

PIXELS_PER_METER = 5
PIXEL_OFFSET = 10  # CAMERA_FORWARD_OFFSET * PIXELS_PER_METER
OFFSET = (-80.0, 160.0)
BIRDVIEW_IMAGE_SIZE = 320
BIRDVIEW_CROP_SIZE = 192
ANGLE_JITTER = 15


def carla_to_opencv(carla_xyz: np.ndarray) -> np.ndarray:
    return np.array([carla_xyz[..., 0], -carla_xyz[..., 2], carla_xyz[..., 1]]).T


def world_coordinate_to_birdview_coordinate(
    location_x: float,
    location_y: float,
    current_location_x: float,
    current_location_y: float,
    current_forward_x: float,
    current_forward_y: float,
):
    """
    Get the birdview coordinate of a world location relative to a current location and
    orientation.

    Based on
    https://github.com/dianchen96/LearningByCheating/blob/031308a77a8ca7e9325ae909ebe04a34105b5d81/bird_view/utils/datasets/image_lmdb.py

    :param location_x: The x-component of a world location.
    :param location_y: The y-component of a world location.
    :param current_location_x: The x-component of the current world location.
    :param current_location_y: The y-component of the current world location.
    :param current_forward_x: The x-component of the vector pointing forwards according
    to the current orientation.
    :param current_forward_y: The y-component of the vector pointing forwards according
    to the current orientation.
    """
    birdview_dx = (location_x - current_location_x) * PIXELS_PER_METER
    birdview_dy = (location_y - current_location_y) * PIXELS_PER_METER

    birdview_x = -birdview_dx * current_forward_y + birdview_dy * current_forward_x
    birdview_y = BIRDVIEW_IMAGE_SIZE - (
        birdview_dx * current_forward_x + birdview_dy * current_forward_y
    )

    birdview_x += OFFSET[1]
    birdview_y += OFFSET[0]

    birdview_x -= (BIRDVIEW_IMAGE_SIZE - BIRDVIEW_CROP_SIZE) // 2
    birdview_y = BIRDVIEW_CROP_SIZE - (BIRDVIEW_IMAGE_SIZE - birdview_y) + 70

    birdview_y += PIXEL_OFFSET
    return birdview_x, birdview_y


def world_coordinate_to_ego_coordinate(
    location_x: float,
    location_y: float,
    current_location_x: float,
    current_location_y: float,
    current_forward_x: float,
    current_forward_y: float,
) -> Tuple[float, float]:
    """
    Get the egocentric top-down coordinate of a world location relative to the current
    location and orientation.

    Carla uses Unreal Engine's left-handed coordinate system:

    (up)
    Z
    ^
    |
    |
    |
    |-------> X (forward)
     \
      \
       \
        >
         Y (right)

    This function transforms coordinates to an egocentric, right-handed, top-down
    coordinate system:

    (forward)
    Y
    ^
    |
    |
    |
    |
    |----------> X (right)

    :param location_x: The x-component of a world location.
    :param location_y: The y-component of a world location.
    :param current_location_x: The x-component of the current world location.
    :param current_location_y: The y-component of the current world location.
    :param current_forward_x: The x-component of the vector pointing forwards according
    to the current orientation.
    :param current_forward_y: The y-component of the vector pointing forwards according
    to the current orientation.
    :return: A tuple of the egocentric top-down X and Y coordinates.
    """
    dx = location_x - current_location_x
    dy = location_y - current_location_y

    x = current_forward_y * dx - current_forward_x * dy
    y = current_forward_y * dy + current_forward_x * dx

    return -x, y


def ego_coordinate_to_world_coordinate(
    egocentric_x: float,
    egocentric_y: float,
    current_location_x: float,
    current_location_y: float,
    current_forward_x: float,
    current_forward_y: float,
) -> Tuple[float, float]:
    """
    :param location_x: The x-component of an egocentric location.
    :param location_y: The y-component of an egocentric location.
    :param current_location_x: The x-component of the current world location.
    :param current_location_y: The y-component of the current world location.
    :param current_forward_x: The x-component of the vector pointing forwards according
    to the current orientation.
    :param current_forward_y: The y-component of the vector pointing forwards according
    to the current orientation.
    :return: A tuple of the world top-down X and Y coordinates.
    """
    egocentric_x *= -1

    dx = current_forward_x * egocentric_y + current_forward_y * egocentric_x
    dy = -current_forward_x * egocentric_x + current_forward_y * egocentric_y

    x = current_location_x + dx
    y = current_location_y + dy

    return x, y


def ego_coordinate_to_image_coordinate(
    egocentric_x: float,
    egocentric_y: float,
    fov: float = 90.0,
    image_width: int = 384,
    image_height: int = 160,
    forward_offset: float = 5.4,
) -> Tuple[float, float]:
    """
    Get the egocentric image (forward-viewing) coordinate of an egocentric top-down
    coordinate.

    :param egocentric_x: The x-component of an egocentric coordinate.
    :param egocentric_y: The y-component of an egocentric coordinate.
    :param fov: The camera horizontal field-of-view in degrees.
    :param image_width: The image width in pixels.
    :param image_height: The image height in pixels.
    :param forward_offset: Relative locations close to 0 are projected outside the
    camera field of view. This constant offset places such locations close to the bottom
    of the frame.
    :return: A tuple of the egocentric image X and Y coordinates (points are the 2D
    projected points where rays shot from the point camera intersect with the ground
    plane).
    """
    egocentric_y += forward_offset - CAMERA_FORWARD_OFFSET

    xyz = np.array([egocentric_x, CAMERA_Z_OFFSET, egocentric_y])

    rotation_vector = np.array([0.0, 0.0, 0.0])
    translation_vector = np.array([0.0, 0.0, 0.0])

    focal_length = image_width / (2 * np.tan(fov / 360.0 * np.pi))

    camera_matrix = np.array(
        [
            [focal_length, 0.0, image_width / 2.0],
            [0.0, focal_length, image_height / 2.0],
            [0.0, 0.0, 1.0],
        ]
    )

    projected = cv2.projectPoints(
        xyz, rotation_vector, translation_vector, camera_matrix, None,
    )
    image_xy = projected[0][0][0]

    return image_xy[0], image_xy[1]


def image_coordinate_to_ego_coordinate(
    image_x: float,
    image_y: float,
    fov: float = 90.0,
    image_width: int = 384,
    image_height: int = 160,
    forward_offset: float = 5.4,
):
    """
    Project from image coordinates to world coordinates. This projection assumes the
    coordinate is at the world ground (world coordinate Y = 0), and that ground is flat
    and parallel to the camera.

    :param image_x: The x-component of the image coordinate.
    :param image_y: The y-component of the image coordinate.
    :param fov: The camera horizontal field-of-view in degrees.
    :param image_width: The image width in pixels.
    :param image_height: The image height in pixels.
    :param forward_offset: The constant offset used when projecting using
    `ego_coordinate_to_image_coordinate`.
    :return: A tuple of the egocentric world X and Y coordinates (X is lateral, Y is
    longitudinal).
    """
    central_x = image_width / 2.0
    central_y = image_height / 2.0

    focal_length = image_width / (2 * np.tan(fov / 360.0 * np.pi))

    x = (image_x - central_x) / focal_length
    y = (image_y - central_y) / focal_length

    world_z = 0.0
    world_y = CAMERA_Z_OFFSET / y
    world_x = world_y * x

    return world_x, world_y - forward_offset + CAMERA_FORWARD_OFFSET

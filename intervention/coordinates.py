import numpy as np
import cv2

CAMERA_HEIGHT = 1.4
CAMERA_Y_OFFSET = 2.0

PIXELS_PER_METER = 5
PIXEL_OFFSET = 10  # CAMERA_Y_OFFSET * PIXELS_PER_METER
OFFSET = (-80.0, 160.0)
BIRDVIEW_IMAGE_SIZE = 320
BIRDVIEW_CROP_SIZE = 192
ANGLE_JITTER = 15


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


def world_coordinate_to_image_coordinate(
    location_x: float,
    location_z: float,
    current_location_x: float,
    current_location_z: float,
    current_forward_x: float,
    current_forward_z: float,
    fov: float = 90.0,
    image_width: int = 384,
    image_height: int = 160,
    forward_offset: float = 4.8,
):
    """
    Get the egocentric (forward-viewing) coordinate of a world location relative to the a
    current location and orientation.

    :param location_x: The x-component of a world location.
    :param location_z: The z-component of a world location.
    :param current_location_x: The x-component of the current world location.
    :param current_location_z: The z-component of the current world location.
    :param current_forward_x: The x-component of the vector pointing forwards according
    to the current orientation.
    :param current_forward_z: The z-component of the vector pointing forwards according
    to the current orientation.
    :param fov: The camera field-of-view.
    :param image_width: The image width in pixels.
    :param image_height: The image height in pixels.
    :param forward_offset: Relative locations close to 0 are projected outside the
    camera field of view. This constant offset places such locations close to the bottom
    of the frame.
    """
    dx = location_x - current_location_x
    dz = location_z - current_location_z

    x = -current_forward_z * dx + current_forward_x * dz
    z = current_forward_z * dz + current_forward_x * dx + 4.8

    xyz = np.array([x, CAMERA_Y_OFFSET, z])

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
    # current_forward_x: float,
    # current_forward_y: float,
    fov: float = 90.0,
    image_width: int = 384,
    image_height: int = 160,
    forward_offset: float = 4.8,
):
    """
    Project from image coordinates to world coordinates. This projection assumes the
    coordinate is at the world ground (world coordinate Y = 0), and that ground is flat
    and parallel to the camera.

    :param image_x: The x-component of the image coordinate.
    :param image_y: The y-component of the image coordinate.
    :param fov: The camera field-of-view.
    :param image_width: The image width in pixels.
    :param image_height: The image height in pixels.
    :param forward_offset: The constant offset used when projecting using
    `world_coordinate_to_image_coordinate`.
    """
    central_x = image_width / 2.0
    central_y = image_height / 2.0

    focal_length = image_width / (2 * np.tan(fov / 360.0 * np.pi))

    x = (image_x - central_x) / focal_length
    y = (image_y - central_y) / focal_length

    world_y = 0.0
    world_z = CAMERA_Y_OFFSET / y
    world_x = world_z * x

    return world_x, world_z - forward_offset

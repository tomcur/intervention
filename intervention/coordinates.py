PIXEL_OFFSET = 10
PIXELS_PER_METER = 5
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

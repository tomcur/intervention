"""
This module is separate from ./image.py, so as to be able to import that module without
having a dependency on the Carla API package (such a dependency is unnecessary in e.g.
the offline training code).
"""

import numpy as np
import carla


def carla_image_to_np(carla_img: carla.Image) -> np.ndarray:
    carla_img.convert(carla.ColorConverter.Raw)

    img = np.frombuffer(carla_img.raw_data, dtype=np.dtype(np.uint8))
    img = np.reshape(img, (carla_img.height, carla_img.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]

    return img

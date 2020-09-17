import numpy as np
import carla


def carla_image_to_np(carla_img: carla.Image) -> np.ndarray:
    carla_img.convert(carla.ColorConverter.Raw)

    img = np.frombuffer(carla_img.raw_data, dtype=np.dtype(np.uint8))
    img = np.reshape(img, (carla_img.height, carla_img.width, 4))
    img = img[:, :, :3]
    img = img[:, :, ::-1]

    return img


def buffer_to_np(
    buf: bytes, image_width: int = 384, image_height: int = 160,
) -> np.ndarray:
    img = np.frombuffer(buf, dtype=np.dtype(np.uint8))
    return np.reshape(img, (image_height, image_width, 3))

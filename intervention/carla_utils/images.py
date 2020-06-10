import numpy as np

from carla import ColorConverter

def carla_image_to_np(carla_img):
    carla_img.convert(ColorConverter.Raw)

    img = np.frombuffer(carla_img.raw_data, dtype=np.dtype('uint8'))
    img = np.reshape(img, (carla_img.height, carla_img.width, 4))
    img = img[:,:,:3]
    img = img[:,:,::-1]

    return img

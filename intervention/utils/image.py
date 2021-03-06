from typing import BinaryIO

import numpy as np
from PIL import Image


def buffer_to_np(
    buf: bytes, image_width: int = 384, image_height: int = 160,
) -> np.ndarray:
    img = np.frombuffer(buf, dtype=np.dtype(np.uint8))
    return np.reshape(img, (image_height, image_width, 3))


def image_file_to_np(fp: BinaryIO) -> np.ndarray:
    im = Image.open(fp, formats=("PNG",))
    return np.asarray(im)

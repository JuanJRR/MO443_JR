import logging

import numpy as np

from src.utilities.benchmarking import benchmarking

logger = logging.getLogger(__name__)


@benchmarking
def quantification(imagen, level: int):
    """Reduces the number of intensity levels in an image (Quantization).

    This function transforms an 8-bit image by mapping its original 256
    intensity levels into a smaller, discrete set of levels defined by
    the 'level' parameter. It calculates the width of the intensity
    intervals and redistributes pixel values accordingly.

    Args:
        imagen (numpy.ndarray): The input image array to be quantized.
        level (int): The desired number of discrete intensity levels
            (e.g., 2, 4, 8, 16, etc.).

    Returns:
        numpy.ndarray: The quantized image in uint8 format.
    """
    wide_interval = 256 // level
    transformation = imagen.copy()
    transformation = (transformation // wide_interval) * wide_interval
    transformation = transformation.astype(np.uint8)

    return transformation

import logging

import numpy as np

from src.utilities.benchmarking import benchmarking

logger = logging.getLogger(__name__)


@benchmarking
def combination_images(imagen_a, imagen_b, factor: list[float, float] = None):
    """Performs a linear combination (blending) of two input images.

    This function merges two images by applying a specific weight to each 
    one. The intensities are normalized, scaled by their respective 
    factors, and then summed to produce a composite image.

    Args:
        imagen_a (numpy.ndarray): The first input image array.
        imagen_b (numpy.ndarray): The second input image array. Must have 
            the same dimensions as 'imagen_a'.
        factor (list[float, float], optional): A list containing two 
            scaling factors [weight_a, weight_b]. For standard alpha 
            blending, the sum of these factors is usually 1.0. Defaults to None.

    Returns:
        numpy.ndarray: he resulting blended image as a uint8 array, 
            clipped to the [0, 255] range to prevent overflow.
    """
    imagen_a = imagen_a / 255.0
    imagen_b = imagen_b / 255.0

    imagen_a = imagen_a * factor[0]
    imagen_b = imagen_b * factor[1]

    transformation = (imagen_a + imagen_b) * 255
    transformation = np.clip(transformation, 0, 255)
    transformation = transformation.astype(np.uint8)

    return transformation

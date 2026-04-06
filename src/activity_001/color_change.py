import logging

import numpy as np

from src.utilities.benchmarking import benchmarking

logger = logging.getLogger(__name__)


@benchmarking
def color_change_filter(imagen):
    """Applies a color transformation matrix to an RGB image.

    This function uses a 3x3 transformation matrix (typically for sepia effects) 
    and performs a linear combination of the R, G, and B channels using 
    matrix multiplication (dot product).

    Args:
        imagen (numpy.ndarray): The input color image in RGB format.

    Returns:
        numpy.ndarray: The color-transformed image as a uint8 array.

    Raises:
        ValueError: If the input image is not in 3D (color) format.
    """
    try:
        matrix = np.array(
            [
                [0.393, 0.769, 0.189],
                [0.349, 0.686, 0.168],
                [0.272, 0.534, 0.131],
            ]
        )

        imagen = imagen / 255.0

        transformation = np.matmul(imagen, matrix.T)
        transformation = transformation * 255
        transformation = np.clip(transformation, 0, 255)
        transformation = transformation.astype(np.uint8)

        del imagen
        return transformation
    except ValueError:
        logger.error("The image must be in color (RGB)")


@benchmarking
def color_change_operation(imagen):
    """Performs a per-channel arithmetic color transformation.

    This function manually calculates new Prime values for Red, Green, and Blue 
    channels based on weighted sums of the original components. It is 
    mathematically equivalent to a sepia filter but implemented through 
    explicit channel indexing.

    Args:
        imagen (numpy.ndarray): The input color image in RGB format.

    Returns:
        numpy.ndarray: The transformed image with re-calculated RGB bands.
    """
    try:
        imagen = imagen / 255.0

        R = imagen[:, :, 0]
        G = imagen[:, :, 1]
        B = imagen[:, :, 2]

        R_prime = 0.393 * R + 0.769 * G + 0.189 * B
        G_prime = 0.349 * R + 0.686 * G + 0.168 * B
        B_prime = 0.272 * R + 0.534 * G + 0.131 * B

        transformation = np.stack([R_prime, G_prime, B_prime], axis=2)
        transformation = transformation * 255
        transformation = np.clip(transformation, 0, 255)
        transformation = transformation.astype(np.uint8)

        del imagen
        return transformation
    except ValueError:
        logger.error("The image must be in color (RGB)")

@benchmarking
def color_change_only_band(imagen):
    """Converts an RGB image to a single-band grayscale representation.

    It uses the standard luminance formula (Y' = 0.2989R + 0.5870G + 0.1140B) 
    to reduce a three-channel color image into a single-channel intensity 
    map, reflecting human perception of brightness.

    Args:
        imagen (numpy.ndarray): The input color image in RGB format.

    Returns:
        numpy.ndarray: A 2D grayscale image array.
    """
    try:
        imagen = imagen / 255.0

        R = imagen[:, :, 0]
        G = imagen[:, :, 1]
        B = imagen[:, :, 2]

        transformation = 0.2989 * R + 0.5870 * G + 0.1140 * B
        transformation = transformation * 255
        transformation = np.clip(transformation, 0, 255)
        transformation = transformation.astype(np.uint8)

        del imagen
        return transformation
    except ValueError:
        logger.error("The image must be in color (RGB)")


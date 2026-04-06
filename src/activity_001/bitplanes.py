import logging

import numpy as np

from src.utilities.benchmarking import benchmarking

logger = logging.getLogger(__name__)


@benchmarking
def bitplanes(imagen):
    """Performs bit-plane slicing on an 8-bit grayscale image.

    This function decomposes an image into its eight constituent binary planes. 
    In an 8-bit image, each pixel intensity is represented by a byte; this 
    method extracts each bit position (from bit 0 to bit 7) across the entire 
    image to analyze the contribution of specific bit levels to the overall 
    visual information.

    Args:
        imagen (numpy.ndarray): The input grayscale image array (must be 
            convertible to uint8).

    Returns:
        numpy.ndarray: A 3D array of shape (8, height, width) containing the 
            eight binary planes. Each plane is scaled to [0, 255] for 
            immediate visualization.
    """
    imagen = imagen.astype(np.uint8)

    bits = np.arange(8).reshape(8, 1, 1)
    transformation = (imagen >> bits) & 1
    transformation = (transformation * 255).astype(np.uint8)

    return transformation

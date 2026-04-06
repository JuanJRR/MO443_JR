import logging

from src.utilities.benchmarking import benchmarking

logger = logging.getLogger(__name__)


@benchmarking
def binarization_threshold(imagen, threshold: int = 255):
    """Performs image binarization based on a global threshold.

    This function converts an image (usually grayscale) into a binary image.

    Pixels with an intensity greater than the specified threshold are set to white (255), while the rest are set to black (0).

        Args:
            imagen (numpy.ndarray): The arrangement of the input image to be processed.
            threshold (int, optional): The threshold value for segmentation. Values above this limit are turned blank. Defaults to 255.

        Returns:
            numpy.ndarray: The resulting binarized image is composed solely of 0 and 255 values.
    """
    transformation = imagen.copy()
    transformation = (transformation > threshold) * 255

    return transformation

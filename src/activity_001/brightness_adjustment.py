import logging

from src.utilities.benchmarking import benchmarking

logger = logging.getLogger(__name__)


@benchmarking
def brightness_adjustment(imagen, gamma: float = 1):
    """Performs non-linear brightness adjustment using gamma correction.

    This function transforms pixel intensity using a power law. It is a fundamental technique in digital image processing for correcting exposure problems or improving contrast in dark areas without oversaturating bright areas.

        Args:
            imagen (numpy.ndarray): The arrangement of the input image
            gamma (float, optional): The value of the exponent for the correction.
             - If gamma > 1: The image becomes brighter (useful for shadow detail).
            - If gamma < 1: The image becomes darker.
            - If gamma = 1: The image remains unchanged.
            Defaults to 1.

        Returns:
            numpy.ndarray: The image with the brightness adjusted
    """
    img = imagen / 255.0
    transformation = img ** (1 / gamma)
    transformation = transformation * 255

    del img
    return transformation

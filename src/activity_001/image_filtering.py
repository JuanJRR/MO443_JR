import numpy as np

from src.utilities.benchmarking import benchmarking
from src.utilities.filter import filter

fiters = {
    "h1": np.array(
        [
            [0, 0, -1, 0, 0],
            [0, -1, -2, -1, 0],
            [-1, -2, 16, -2, -1],
            [0, -1, -2, -1, 0],
            [0, 0, -1, 0, 0],
        ]
    ),
    "h2": np.array(
        [
            [1, 4, 6, 4, 1],
            [4, 16, 24, 16, 4],
            [6, 24, 36, 24, 6],
            [4, 16, 24, 16, 4],
            [1, 4, 6, 4, 1],
        ]
    )
    / 256,
    "h3": np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]),
    "h4": np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]),
    "h5": np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
    "h6": np.ones((3, 3)) / 9,
    "h7": np.array([[-1, -1, 2], [-1, 2, -1], [2, -1, -1]]),
    "h8": np.array([[2, -1, -1], [-1, 2, -1], [-1, -1, 2]]),
    "h9": np.eye(9) / 9,
    "h10": np.array(
        [
            [-1, -1, -1, -1, -1],
            [-1, 2, 2, 2, -1],
            [-1, 2, 8, 2, -1],
            [-1, 2, 2, 2, -1],
            [-1, -1, -1, -1, -1],
        ]
    ),
    "h11": np.array([[-1, -1, 0], [-1, 0, 1], [0, 1, 1]]),
}


@benchmarking
def imagen_filtering(imagen):
    """Applies a predefined suite of spatial filters to an input image.

    This function iterates through a dictionary of kernels (h1 to h11) and
    applies each one using a convolution operation. Additionally, it
    computes a derived filter (h12) representing the gradient magnitude
    using the Sobel operators (h3 and h4).

    Args:
        imagen (numpy.ndarray): The input image array (grayscale or RGB).

    Returns:
        dict: A dictionary where keys are filter identifiers (e.g., 'h1',
            'h2', ..., 'h12') and values are the resulting transformed
            image arrays in uint8 format.
    """
    transformations = {}
    for id, kernel in fiters.items():
        transformations[id] = filter(imagen=imagen, kernel=kernel)

    h12 = np.sqrt(transformations["h3"] ** 2 + transformations["h4"] ** 2)
    h12 = (np.clip(h12, 0, 255)).astype(np.uint8)
    transformations["h12"] = h12

    return transformations

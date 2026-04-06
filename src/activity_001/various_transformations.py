import logging

import numpy as np

from src.utilities.benchmarking import benchmarking

logger = logging.getLogger(__name__)

@benchmarking
def reverse(imagen):
    """Calculates the photographic negative of an image.

    Args:
        imagen (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Inverted image in uint8 format.
    """
    transformation = imagen.copy()
    transformation = 255-transformation
    transformation = np.clip(transformation, 0, 255).astype(np.uint8)
    
    return transformation

@benchmarking
def intensity_rescaling(imagen):
    """Linearly rescales image intensities to a specific range [100, 200].

    This transformation maps the original minimum and maximum values 
    to a new set of bounds, useful for normalizing contrast across 
    different images.

    Args:
        imagen (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Rescaled image with intensities between 100 and 200.
    """
    transformation = imagen.copy()
    
    transformation = transformation.astype(np.float32)
    img_min, img_max = transformation.min(), transformation.max()

    if img_max != img_min:
        transformation = (transformation - img_min) * (200 - 100) / (img_max - img_min) + 100
    else:
        transformation = np.full_as(transformation, 150)


    transformation = transformation.astype(np.uint8)
    

    return transformation

@benchmarking
def inverted_even_lines(imagen):
    """Horizontally flips every even-indexed row in the image.

    This creates a specialized visual distortion effect by reversing the 
    pixel order of alternating lines.

    Args:
        imagen (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Image with alternating horizontally flipped lines.
    """
    transformation = imagen.copy()
    transformation[::2, :] = transformation[::2, ::-1]
    transformation = transformation.astype(np.uint8)
    
    return transformation

@benchmarking
def reflection_upper_half(imagen):
    """Reflects the top half of the image onto the bottom half.

    The function takes the upper 50% of the image and overwrites the 
    lower 50% with a vertically flipped version of the top.

    Args:
        imagen (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Vertically symmetrical image based on the top half.
    """
    transformation = imagen.copy()

    half = transformation.shape[0] // 2
    transformation[half:] = transformation[:half][::-1]
    transformation = transformation.astype(np.uint8)

    return transformation

@benchmarking
def vertical_reflection(imagen):
    """Flips the entire image vertically (upside down).

    Args:
        imagen (numpy.ndarray): Input image array.

    Returns:
        numpy.ndarray: Vertically flipped image.
    """
    transformation = imagen.copy()
    transformation = transformation[::-1, :]
    transformation = transformation.astype(np.uint8)

    return transformation

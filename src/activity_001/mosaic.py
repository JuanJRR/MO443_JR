import logging

import numpy as np

from src.utilities.benchmarking import benchmarking

logger = logging.getLogger(__name__)


@benchmarking
def mosaics(imagen, n_blocks:int, new_arrangement:np.ndarray):
    """Divides an image into a grid of square blocks and rearranges them.

    This function reshapes the input image into a multi-dimensional grid of 
    size (n_blocks x n_blocks). It then reorders these blocks according to 
    the provided mapping and reconstructs the image back to its original 
    spatial dimensions.

    Args:
        imagen (numpy.ndarray): The input image array (supports Grayscale 2D 
            or RGB 3D).
        n_blocks (int): The number of blocks per side (e.g., if n_blocks=3, 
            the image is divided into a 3x3 grid for a total of 9 blocks).
        new_arrangement (np.ndarray): A 1D array of integers 
            representing the new indices for the blocks (0-indexed).

    Returns:
        numpy.ndarray: The transformed image with rearranged blocks, 
            padded and trimmed to ensure exact divisibility by n_blocks.
    """
    height, width = imagen.shape[:2]

    pad_h = (n_blocks - height % n_blocks) % n_blocks
    pad_w = (n_blocks - width % n_blocks) % n_blocks

    if imagen.ndim == 3:
        config_pad = ((0, pad_h), (0, pad_w), (0, 0))
    else:
        config_pad = ((0, pad_h), (0, pad_w))

    imagen = np.pad(imagen, config_pad, mode="edge")
    height, width = imagen.shape[:2]

    height_new = (height // n_blocks) * n_blocks
    width_new = (width // n_blocks) * n_blocks

    img_trimmed = imagen[:height_new, :width_new]

    cell_h = height_new // n_blocks
    cell_w = width_new // n_blocks

    if imagen.ndim == 3:
        shape = (n_blocks, cell_h, n_blocks, cell_w, imagen.shape[2])
        order = (0, 2, 1, 3, 4)
    else:
        shape = (n_blocks, cell_h, n_blocks, cell_w)
        order = (0, 2, 1, 3)

    blocks = img_trimmed.reshape(shape)
    blocks = blocks.transpose(order)

    reordered_mosaic = blocks[new_arrangement // n_blocks, new_arrangement % n_blocks]

    final_shape = (
        (height_new, width_new, 3) if imagen.ndim == 3 else (height_new, width_new)
    )

    transformation = reordered_mosaic.transpose(order)
    transformation = transformation.reshape(final_shape)

    return transformation

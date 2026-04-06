import logging

import numpy as np

logger = logging.getLogger(__name__)


def filter(imagen, kernel: np.ndarray):
    """Applies a spatial linear filter to an image using a convolution kernel.

    This function performs a manual 2D convolution by sliding a kernel over
    the input image. It supports both grayscale (2D arrays) and color
    (3D arrays) images, using 'edge' padding to handle boundary conditions
    without introducing dark artifacts at the borders.

    Args:
        imagen (numpy.ndarray): Input image array. Can be a 2D array
            (grayscale) or a 3D array (RGB).
        kernel (np.ndarray): A square 2D array representing the filter
            (e.g., Sobel, Laplacian, or Box filter). The dimension should
            ideally be odd (3x3, 5x5, etc.).

    Returns:
        numpy.ndarray: The filtered image as an 8-bit unsigned integer array
            (uint8), normalized to the [0, 255] range.
    """

    imagen = imagen / 255.0

    k_dim = kernel.shape[0]
    padding = k_dim // 2
    is_color = imagen.ndim == 3

    pad_width = (
        ((padding, padding), (padding, padding), (0, 0)) if is_color else padding
    )

    img_padding = np.pad(imagen, pad_width, mode="edge")

    y_idx, x_idx = np.indices((k_dim, k_dim))
    y_offset = y_idx.ravel()
    x_offset = x_idx.ravel()

    img_high, img_width = imagen.shape[:2]
    rows, cols = np.indices((img_high, img_width))

    rows_broad = rows[None, :, :] + y_offset[:, None, None]
    cols_broad = cols[None, :, :] + x_offset[:, None, None]

    neighbors = img_padding[rows_broad, cols_broad]

    transformation = np.tensordot(kernel.ravel(), neighbors, axes=(0, 0))

    t_max = transformation.max()
    if t_max > 0:
        transformation = transformation / t_max

    transformation = (transformation * 255).astype(np.uint8)

    return transformation

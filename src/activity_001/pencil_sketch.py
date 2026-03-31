import logging

import cv2
import numpy as np

from src.utilities.benchmarking import benchmarking

logger = logging.getLogger(__name__)


def __kernel_gaussian(dim_k: int = 3, sigma: float = 0.15):
    """Generates a matrix (kernel) based on the two-dimensional normal distribution.

    Args:
        dim_k (int, optional): Square dimension of the Gaussian kernel. Defaults to 3.
        sigma (float, optional): Standard deviation value given a mean equal to zero. Defaults to 0.46.

    Returns:
        numpy.ndarray: Returns a kernel matrix dim_k x dim_k, under a Gaussian distribution.
    """
    points = np.linspace(-1, 1, dim_k)
    width_k, high_k = np.meshgrid(points, points)

    kernel = ((1) / (2 * np.pi * (sigma**2))) * np.exp(
        -(((width_k**2) + (high_k**2)) / (2 * (sigma**2)))
    )

    return kernel


def __gaussian_filter(imagen, kernel):
    """Implementation of the layer displacement filtering technique (Stacking or im2col)

    Args:
        imagen (numpy.ndarray): Image to apply filter to
        kernel (numpy.ndarray): Filter kernel

    Returns:
        _type_: Returns an array with the application of the provided kernel
    """

    k_dim = kernel.shape[0]
    padding = k_dim // 2
    img_padding = np.pad(imagen, padding, mode="edge")

    y_idx, x_idx = np.indices((k_dim, k_dim))
    y_offset = y_idx.ravel()
    x_offset = x_idx.ravel()

    img_high, img_widh = imagen.shape
    rows, cols = np.indices((img_high, img_widh))

    rows_broad = rows[None, :, :] + y_offset[:, None, None]
    cols_broad = cols[None, :, :] + x_offset[:, None, None]

    neighbors = img_padding[rows_broad, cols_broad]

    transformation = np.tensordot(np.ravel(kernel), neighbors, axes=(0, 0))
    transformation = transformation / transformation.max()

    del neighbors, img_padding
    return transformation


@benchmarking
def pencil_sketch(imagen, dim_k: int = 3, sigma: float = 1):
    """Applies a pencil sketch effect using manual Gaussian filtering and Color Dodge.

    Args:
        imagen (_type_): The input grayscale image.
        dim_k (int, optional): Kernel size for the Gaussian filter. Defaults to 3.
        sigma (float, optional): Standard deviation for Gaussian distribution. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - transformation (numpy.ndarray): The resulting pencil sketch image.
            - blur (numpy.ndarray): The intermediate blurred image for analysis.
    """
    imagen = imagen.astype(float) / 255.0
    img_inv = 1.0 - imagen

    kernel = __kernel_gaussian(dim_k=dim_k, sigma=sigma)
    blur = __gaussian_filter(imagen=img_inv, kernel=kernel)

    transformation = imagen / (1 - (blur + 0.0000001))
    transformation = np.clip(transformation, 0.0, 1.0)
    transformation = (transformation * 255).astype(np.uint8)

    del imagen, img_inv, kernel
    return transformation, blur


@benchmarking
def pencil_sketch_cv(imagen, dim_k: int = 3, sigma: float = 1):
    """Applies a pencil sketch effect using optimized OpenCV

    Args:
        imagen (numpy.ndarray): Input grayscale image
        dim_k (int, optional): Kernel size. Defaults to 3.
        sigma (float, optional): Gaussian sigma. Defaults to 1.

    Returns:
        tuple: A tuple containing:
            - transformation (numpy.ndarray): The resulting pencil sketch image.
            - blur (numpy.ndarray): The intermediate blurred image for analysis.
    """
    imagen = imagen.astype(float) / 255.0
    img_inv = 1.0 - imagen

    kernel = __kernel_gaussian(dim_k=dim_k, sigma=sigma)
    blur = cv2.filter2D(src=img_inv, ddepth=-1, kernel=kernel)
    blur = blur / blur.max()

    transformation = imagen / (1 - (blur + 0.0000001))
    transformation = np.clip(transformation, 0.0, 1.0)
    transformation = (transformation * 255).astype(np.uint8)

    del imagen, img_inv, kernel
    return transformation, blur

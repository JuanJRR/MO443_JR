import cv2
import numpy as np

from src.utilities.benchmarking import benchmarking


@benchmarking
def image_scaling(imagen, factor):
    """Performs image scaling using a manual nearest-neighbor interpolation approach.

    This function implements the geometric transformation of scaling by mapping 
    destination coordinates back to the source image (inverse mapping). This 
    method ensures that every pixel in the new grid is assigned a value, 
    preventing "holes" in the output.

    Args:
        imagen (numpy.ndarray): The source image to be scaled.
        factor (_type_): The scaling multiplier (e.g., 0.5 for half-size, 
            2.0 for double-size).

    Returns:
        numpy.ndarray: The scaled image with new dimensions (height * factor, 
            width * factor).
    """
    height, width = imagen.shape[:2]
    high_scale = int(height * factor)
    width_scale = int(width * factor)

    y_dest, x_dest = np.indices((high_scale, width_scale))
    coords_dest = np.vstack((x_dest.ravel(), y_dest.ravel()))

    coords_orig = coords_dest / factor

    x_orig = np.round(coords_orig[0, :]).astype(int)
    y_orig = np.round(coords_orig[1, :]).astype(int)

    x_orig = np.clip(x_orig, 0, width - 1)
    y_orig = np.clip(y_orig, 0, height - 1)

    shape_dest = (high_scale, width_scale) + imagen.shape[2:]
    image_scaled = np.zeros(shape_dest, dtype=imagen.dtype)

    image_scaled[y_dest.ravel(), x_dest.ravel()] = imagen[y_orig, x_orig]

    return image_scaled

@benchmarking
def image_scaling_cv(imagen, factor:float):
    """Scales an image using OpenCV's highly optimized 'resize' function.

    This serves as the performance baseline to compare against the manual 
    implementation. By default, it uses bilinear interpolation (OpenCV standard).

    Args:
        imagen (numpy.ndarray): The source image.
        factor (float): The scaling multiplier.

    Returns:
        numpy.ndarray: The scaled image processed by OpenCV's internal kernels.
    """
    height, width = imagen.shape[:2]
    high_scale = int(height * factor)
    width_scale = int(width * factor)

    image_scaled = cv2.resize(imagen, (width_scale, high_scale))

    return image_scaled

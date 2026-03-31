import logging

import cv2
import numpy as np

from src.utilities.benchmarking import benchmarking

logger = logging.getLogger(__name__)


@benchmarking
def image_rotation(image, degrees: float):
    """Rotates an image by a given angle using a manual transformation matrix.

    This implementation performs a forward mapping rotation by calculating the
    new coordinates for each pixel using a 2D rotation matrix.

    Args:
        image (numpy.ndarray): The input image array to be rotated.
        degrees (float): Rotation angle in degrees (clockwise).

    Returns:
        numpy.ndarray: The resulting rotated image with the same dimensions
            as the input.

    Note:
        - The rotation is performed around the center of the image.
        - Pixels that fall outside the original boundaries after rotation
          are clipped.
        - This manual method is monitored by the @benchmarking decorator to
          analyze its computational cost.
    """
    theta = -degrees * (np.pi / 180)

    res_shape = image.shape
    height, width = res_shape[:2]
    rotation_point_y = width / 2
    rotation_point_x = height / 2

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    rotation_matrix = np.array(
        [
            [cos_theta, -sin_theta],
            [sin_theta, cos_theta],
        ]
    )

    y_dest, x_dest = np.indices((height, width))
    y_coords = y_dest.ravel() - rotation_point_y
    x_coords = x_dest.ravel() - rotation_point_x

    coords_dest = np.vstack((x_coords, y_coords))
    coords_orig = np.matmul(rotation_matrix, coords_dest)

    x_orig = (coords_orig[0, :] + rotation_point_x).astype(int)
    y_orig = (coords_orig[1, :] + rotation_point_y).astype(int)

    mask = (x_orig >= 0) & (x_orig < width) & (y_orig >= 0) & (y_orig < height)

    rotated_image = np.zeros(res_shape, dtype=image.dtype)
    rotated_image[y_dest.ravel()[mask], x_dest.ravel()[mask]] = image[
        y_orig[mask], x_orig[mask]
    ]

    return rotated_image


@benchmarking
def image_rotation_cv(image, degrees: float):
    """Rotates an image using OpenCV's optimized affine transformation.

    This function serves as a high-performance alternative to the manual 
    rotation, utilizing OpenCV's `getRotationMatrix2D` and `warpAffine` 
    functions.

    Args:
        image (numpy.ndarray): The input image array.
        degrees (float): Rotation angle in degrees.

    Returns:
        numpy.ndarray: The rotated image processed by OpenCV.

    Note:
        - Includes basic error handling to check if the image is valid before 
          processing.
        - Highly recommended for TPs requiring multiple geometric 
          transformations where execution time is critical.
    """
    if image is None:
        logging.error("Could not read image file. Check the path.")
    else:
        height, width = image.shape[:2]
        center = [width / 2, height / 2]
        scale = 1.0

        trsf = cv2.getRotationMatrix2D(center, -degrees, scale)
        rotated_image = cv2.warpAffine(image, trsf, (width, height))

        return rotated_image

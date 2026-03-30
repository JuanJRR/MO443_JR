import os

import cv2


def upload_images(path: str, color: bool = False):
    """
    Loads an image from a file path and handles color space conversion.

    By default, OpenCV reads images in BGR format. This function converts
    them to RGB for color processing or to Grayscale for classical image
    analysis.

    Args:
        path (str): The file system path to the image to be loaded.
        color (bool, optional): If True, converts the image from BGR to RGB
            format. If False, converts the image to Grayscale. Defaults to False.

    Returns:
        numpy.ndarray: The processed image array in the requested color space.

    Note:
        Grayscale conversion is particularly useful for initial TPs covering
        histogram transformations and spatial filtering.
    """
    image = cv2.imread(path)

    if color:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    return image


def save_images(image, path: str, name_save: str = "original_image.png"):
    """Saves an image array to a specified file path.

    This utility uses OpenCV's `imwrite` function to export image data from 
    memory to a persistent file.

    Args:
        image (numpy.ndarray): The image array to be saved.
        path (str): The directory path where the image will be stored.
        name_save (str, optional): The name of the output file, including 
            its extension (e.g., 'result.png'). Defaults to "original_image".
    """
    cv2.imwrite(os.path.join(path, name_save), image)

import logging

from src.activity_001.rotation import image_rotation, image_rotation_cv
from src.activity_001.scaling import image_scaling, image_scaling_cv
from src.utilities.graphics import Graphics
from src.utilities.load_save import save_images, upload_images

logger = logging.getLogger(__name__)

views = Graphics()


# Image rotation
def rotation_analysis(path: str):
    """Executes a complete rotation analysis workflow on a target image.

    This function performs a series of rotations (90, 180, 270, 360 degrees)
    using both the manual implementation and the OpenCV-based version.
    It logs the progress, saves the raw results, and generates comparative
    visualizations for each step.

    Args:
        path (str): File path to the input image for the rotation test.

    Note:
        - Results are stored in '/app/experiments/activity_001/results/rotation'.
    """
    logger.info("Start image rotation transformation.")

    path_save = "/app/experiments/activity_001/results/rotation"

    img = upload_images(path=path, color=False)
    views.view_simplified(
        image=img,
        title="Imagem Original",
        save=True,
        path_save=path_save,
        name_save="original_image",
        plot=False,
    )

    for degree in [90, 180, 270, 360]:
        logger.info(f"Transformation to {degree} degrees with proposed implementation")

        rotated_image = image_rotation(image=img, degrees=degree)
        save_images(
            image=rotated_image,
            path=path_save,
            name_save=f"{degree}_degree_image_raw.png",
        )
        views.view_simplified(
            image=rotated_image,
            title=f"Imagem a {degree} Graus (Implementação)",
            save=True,
            path_save="/app/experiments/activity_001/results/rotation",
            name_save=f"{degree}_degree_image",
            plot=False,
        )

        del rotated_image

        logger.info(
            f"Transformation to {degree} degrees with implementation provided by OpenCV."
        )

        rotated_image_cv = image_rotation_cv(image=img, degrees=degree)
        save_images(
            image=rotated_image_cv,
            path=path_save,
            name_save=f"{degree}_degree_image_cv_raw.png",
        )
        views.view_simplified(
            image=rotated_image_cv,
            title=f"Imagem a {degree} Graus (OpenCV)",
            save=True,
            path_save="/app/experiments/activity_001/results/rotation",
            name_save=f"{degree}_degree_image_cv",
            plot=False,
        )

        del rotated_image_cv

    logger.info("Completion of rotation transformation")


# image scaling
def scaling_analysis(path: str):
    """Run a scaling analysis workflow to evaluate size transformations.

    This function tests the image's scalability by applying a factor of (2, 4, 6), using both a custom nearest neighbor implementation and OpenCV's resizing method. It generates side-by-side visual comparisons to analyze quality.

        Args:
            path (str): File path to the input image.

        Note:
            - Results are stored in '/app/experiments/activity_001/results/scaling'.
    """
    logger.info("Start image scaling transformation.")

    path_save = "/app/experiments/activity_001/results/scaling"

    img = upload_images(path=path, color=False)
    views.view_simplified(
        image=img,
        title="Imagem Original",
        save=True,
        path_save=path_save,
        name_save="original_image",
        plot=False,
    )

    for scaling in [2, 4, 6]:
        logger.info(
            f"Transformation to scaling {scaling}x with proposed implementation"
        )

        scaling_image = image_scaling(imagen=img, factor=scaling)

        save_images(
            image=scaling_image,
            path=path_save,
            name_save=f"image_scaled_{scaling}x_raw.png",
        )

        views.view_comparison(
            img_orig=img,
            img_trand=scaling_image,
            factor=scaling,
            info_trasd=f"Transformação de Ampliação de {scaling}X (Implementação)",
            save=True,
            path_save=path_save,
            name_save=f"image_scaled_{scaling}x",
            plot=False,
            bar=False,
        )

        del scaling_image

        logger.info(
            f"Transformation to scaling {scaling}x with implementation provided by OpenCV."
        )

        scaling_image_cv = image_scaling_cv(imagen=img, factor=scaling)

        save_images(
            image=scaling_image_cv,
            path=path_save,
            name_save=f"image_scaled_{scaling}x_cv_raw.png",
        )

        views.view_comparison(
            img_orig=img,
            img_trand=scaling_image_cv,
            factor=scaling,
            info_trasd=f"Transformação de Ampliação de {scaling}X (OpenCV)",
            save=True,
            path_save=path_save,
            name_save=f"image_scaled_{scaling}x_cv",
            plot=False,
            bar=False,
        )

        del scaling_image_cv

    logger.info("Completion of scaling transformation")

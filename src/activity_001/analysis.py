import logging

from src.activity_001.binarization_threshold import binarization_threshold
from src.activity_001.brightness_adjustment import brightness_adjustment
from src.activity_001.pencil_sketch import pencil_sketch, pencil_sketch_cv
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

    del img, path_save
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

    del img, path_save
    logger.info("Completion of scaling transformation")


# imagen pencil sketch
def pencil_sketch_analysis(path: str):
    """Executes a comprehensive analysis of the pencil sketch effect.

    This function iterates through different combinations of Gaussian kernel
    sizes and sigma values to evaluate the 'Pencil Sketch' transformation.
    It compares the manual implementation against the OpenCV optimized version,
    generating visual reports that include the original image, the intermediate
    blurred version, and the final artistic result.

    Args:
        path (str): File path to the source image to be processed.

    Note:
        - Results are organized in '/app/experiments/activity_001/results/pencil_sketch'.
        - For each parameter set, a three-panel comparison is generated using
          'view_comparison_mult'.
        - The analysis demonstrate the effect of different scales
          of blur on the sketch outlines.
    """
    logger.info("Start image pencil sketch transformation.")

    path_save = "/app/experiments/activity_001/results/pencil_sketch"

    img = upload_images(path=path, color=False)
    views.view_simplified(
        image=img,
        title="Imagem Original",
        save=True,
        path_save=path_save,
        name_save="original_image",
        plot=False,
    )
    for kernel in [3, 11, 21]:
        for sigma in [0.15, 0.25, 0.35, 1]:
            transformation, blur = pencil_sketch(imagen=img, dim_k=kernel, sigma=sigma)

            views.view_comparison_mult(
                img_orig=img,
                img_trand_1=blur,
                img_trand_2=transformation,
                info_trasd_1=f"Desfoque Gaussiano [$\sigma={sigma}; k={kernel}x{kernel}$] \n(Implementação)",
                info_trasd_2="Transformação do Esboço a Lápis \n(Implementação)",
                save=True,
                path_save=path_save,
                name_save=f"pencilsketch_s{str(sigma).replace('.', '')}_k{kernel}",
                bar=True,
                plot=False,
            )

            save_images(
                image=transformation,
                path=path_save,
                name_save=f"pencilsketch_s{str(sigma).replace('.', '')}_k{kernel}_raw.png",
            )

            del transformation, blur

            transformation, blur = pencil_sketch_cv(
                imagen=img, dim_k=kernel, sigma=sigma
            )

            views.view_comparison_mult(
                img_orig=img,
                img_trand_1=blur,
                img_trand_2=transformation,
                info_trasd_1=f"Desfoque Gaussiano [$\sigma={sigma}; k={kernel}x{kernel}$] \n(OpenCV)",
                info_trasd_2="Transformação do Esboço a Lápis \n(OpenCV)",
                save=True,
                path_save=path_save,
                name_save=f"pencilsketch_s{str(sigma).replace('.', '')}_k{kernel}_cv",
                bar=True,
                plot=False,
            )

            save_images(
                image=transformation,
                path=path_save,
                name_save=f"pencilsketch_s{str(sigma).replace('.', '')}_k{kernel}_cv_raw.png",
            )

    del img, path_save
    logger.info("Completion of pencil sketch transformation")


# brightness adjustment
def brightness_adjustment_analysis(path: str):
    """Executes a systematic analysis of non-linear brightness adjustments.

    This function evaluates the Gamma Correction (Power Law Transformation)
    by applying a range of gamma values to a source image. It generates
    individual reports for each gamma value and composite multi-panel
    comparisons to study the effect of the '1/gamma' exponent on image
    luminosity and contrast.

    Args:
        path (str): File path to the source image to be analyzed.

    Note:
        - Results are stored in '/app/experiments/activity_001/results/brightness'.
        - Tested gamma values: [1.5, 2.5, 3.5, 5.5].
        - Uses 'view_comparison' for individual results and 'view_comparison_mult'
          for grouped comparisons.
    """
    logger.info("Start image brightness adjustment transformation.")

    path_save = "/app/experiments/activity_001/results/brightness_adjustment"

    img = upload_images(path=path, color=False)
    views.view_simplified(
        image=img,
        title="Imagem Original",
        save=True,
        path_save=path_save,
        name_save="original_image",
        plot=False,
    )

    transformations = []
    for gamma in [1.5, 2.5, 3.5, 5.5]:
        transformation = brightness_adjustment(imagen=img, gamma=gamma)
        transformations.append(transformation)

        views.view_comparison(
            img_orig=img,
            img_trand=transformation,
            info_trasd=f"Brightness Adjustment ($\gamma = {gamma}$)",
            save=True,
            path_save=path_save,
            name_save=f"brightness_adjustment_g{str(gamma).replace('.', '')}",
            plot=False,
            bar=True,
        )

        save_images(
            image=transformation,
            path=path_save,
            name_save=f"brightness_adjustment_g{str(gamma).replace('.', '')}_raw.png",
        )

        del transformation

    views.view_comparison_mult(
        img_orig=img,
        img_trand_1=transformations[0],
        img_trand_2=transformations[1],
        info_trasd_1="Brightness Adjustment ($\gamma = 1.5$)",
        info_trasd_2="Brightness Adjustment ($\gamma = 2.5$)",
        save=True,
        path_save=path_save,
        name_save="brightness_adjustment_com1",
        bar=True,
        plot=False,
    )

    views.view_comparison_mult(
        img_orig=img,
        img_trand_1=transformations[2],
        img_trand_2=transformations[3],
        info_trasd_1="Brightness Adjustment ($\gamma = 3.5$)",
        info_trasd_2="Brightness Adjustment ($\gamma = 5.5$)",
        save=True,
        path_save=path_save,
        name_save="brightness_adjustment_com2",
        bar=True,
        plot=False,
    )

    del transformations, img, path_save
    logger.info("Completion of brightness adjustment transformation")


# Binarization by threshold
def binarization_threshold_analysis(path: str):
    """Executes a systematic analysis of global image binarization.

    This function evaluates how different global threshold levels affect the 
    segmentation of an image into a binary (black and white) format. It 
    iterates through a set of predefined thresholds, generates 
    visualizations for each, and produces a final multi-panel comparison 
    to analyze the loss of detail versus object isolation.

    Args:
        path (str): File path to the source image (grayscale) to be analyzed.

    Note:
        - Results are organized in '/app/experiments/activity_001/results/binarization_threshold'.
        - Tested threshold levels: [64, 128, 192].
        - Each iteration saves a raw binary image and a comparative plot 
          (Original vs. Binarized).
        - A final comparison is generated using 'view_comparison_mult' to 
          display the results of all three thresholds side-by-side.
    """
    logger.info("Start image binarization by threshold transformation.")

    path_save = "experiments/activity_001/results/binarization_threshold"

    img = upload_images(path=path, color=False)
    views.view_simplified(
        image=img,
        title="Imagem Original",
        save=True,
        path_save=path_save,
        name_save="original_image",
        plot=False,
    )

    thresholds = []
    for threshold in [64, 128, 192]:
        transformation = binarization_threshold(imagen=img, threshold=threshold)
        thresholds.append(transformation)

        save_images(
            image=transformation,
            path=path_save,
            name_save=f"binarized_image_threshold_{threshold}_raw.png",
        )

        views.view_comparison(
            img_orig=img,
            img_trand=transformation,
            info_trasd=f"Imagem Binarizada ($Limiar = {threshold}$)",
            path_save=path_save,
            name_save=f"binarized_image_threshold_{threshold}",
            save=True,
            bar=False,
            plot=False,
        )

        del transformation

    views.view_comparison_mult(
        img_orig=thresholds[0],
        img_trand_1=thresholds[1],
        img_trand_2=thresholds[2],
        img_trand_orig="Imagem Binarizada ($Limiar = 64$)",
        info_trasd_1="Imagem Binarizada ($Limiar = 128$)",
        info_trasd_2="Imagem Binarizada ($Limiar = 192$)",
        path_save=path_save,
        name_save="binarized_image_threshold_comp",
        save=True,
        bar=True,
        plot=False,
    )

    del thresholds, img, path_save
    logger.info("Completion of binarization by threshold transformation")

import logging
import numpy as np

from src.activity_001.bitplanes import bitplanes
from src.activity_001.color_change import color_change_only_band
from src.activity_001.color_change import color_change_operation
from src.activity_001.color_change import color_change_filter
from src.activity_001.mosaic import mosaics
from src.activity_001.binarization_threshold import binarization_threshold
from src.activity_001.brightness_adjustment import brightness_adjustment
from src.activity_001.pencil_sketch import pencil_sketch, pencil_sketch_cv
from src.activity_001.rotation import image_rotation, image_rotation_cv
from src.activity_001.scaling import image_scaling, image_scaling_cv
from src.utilities.graphics import Graphics
from src.utilities.load_save import save_images, upload_images
from src.activity_001.combination_images import combination_images

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


# mosaics
def mosaics_analysis(path: str):
    """Executes an experimental analysis of image mosaicing and block rearrangement.

    This function divides the target image into a 3x3 grid (9 blocks) and
    applies a predefined pseudo-random rearrangement. It validates the
    logic of spatial indexing and block-based reconstruction.

    Args:
        path (str): File path to the source image (supports RGB or Grayscale).

    Note:
        - Results are stored in 'experiments/activity_001/results/mosaic'.
    """
    logger.info("Start image mosaics transformation.")

    path_save = "experiments/activity_001/results/mosaics"

    img = upload_images(path=path, color=False)
    views.view_simplified(
        image=img,
        title="Imagem Original",
        save=True,
        path_save=path_save,
        name_save="original_image",
        plot=False,
    )

    new_mosaic = np.array(
        [
            [6, 11, 13, 3],
            [8, 16, 1, 9],
            [12, 14, 2, 7],
            [4, 15, 10, 5],
        ]
    )
    new_mosaic = new_mosaic - 1

    transformation = mosaics(imagen=img, n_blocks=4, new_arrangement=new_mosaic)

    save_images(
        image=transformation,
        path=path_save,
        name_save="mosaic_raw.png",
    )

    views.view_comparison(
        img_orig=img,
        img_trand=transformation,
        info_trasd="Mosaico ($blocos=4$)",
        path_save=path_save,
        name_save="mosaic_comp",
        save=True,
        bar=False,
        plot=False,
    )

    del transformation, img, new_mosaic
    logger.info("Completion of mosaics transformation")


# color change filter
def color_change_filter_analysis(path: str):
    """Analyzes color transformation using matrix-based filtering.

    Applies a 3x3 transformation matrix (linear combination of RGB channels)
    to the input image. This method is used to evaluate the efficiency and
    visual output of global color filtering via 'np.matmul'.

    Args:
        path (str): File path to the input RGB image.

    Note:
        - Output directory: 'experiments/activity_001/results/color_change'.
    """
    logger.info("Start image color change filter transformation.")

    path_save = "experiments/activity_001/results/color_change"

    img = upload_images(path=path, color=True)
    views.view_simplified(
        image=img,
        title="Imagem Original",
        save=True,
        path_save=path_save,
        name_save="original_image",
        plot=False,
    )

    transformation = color_change_filter(imagen=img)

    save_images(
        image=transformation,
        path=path_save,
        name_save="color_change_filter_raw.png",
    )

    views.view_comparison(
        img_orig=img,
        img_trand=transformation,
        info_trasd="Transformação de cores",
        path_save=path_save,
        name_save="color_change_filter_comp",
        save=True,
        bar=False,
        plot=False,
    )

    del transformation, img
    logger.info("Completion of color change filter transformation")


# color change operation
def color_change_operation_analysis(path: str):
    """Evaluates color transformation through manual channel-wise arithmetic.

    This analysis applies the same color weights as the filter version but
    processes each RGB channel (Red, Green, Blue) explicitly. It is used
    to verify mathematical consistency between matrix operations and
    standard arithmetic.

    Args:
        path (str): File path to the input RGB image.

    Note:
        - Output directory: 'experiments/activity_001/results/color_change'.
    """
    logger.info("Start image color change operation transformation.")

    path_save = "experiments/activity_001/results/color_change"

    img = upload_images(path=path, color=True)
    views.view_simplified(
        image=img,
        title="Imagem Original",
        save=True,
        path_save=path_save,
        name_save="original_image",
        plot=False,
    )

    transformation = color_change_operation(imagen=img)

    save_images(
        image=transformation,
        path=path_save,
        name_save="color_change_operation_raw.png",
    )

    views.view_comparison(
        img_orig=img,
        img_trand=transformation,
        info_trasd="Transformação de cores",
        path_save=path_save,
        name_save="color_change_operation_comp",
        save=True,
        bar=False,
        plot=False,
    )

    del transformation, img
    logger.info("Completion of color change operation transformation")


# color change only band
def color_change_onlyband_analysis(path: str):
    """Analyzes the conversion of an RGB image to a single-channel grayscale band.

    Uses a weighted average of RGB channels based on human luminance perception
    (Y' = 0.2989R + 0.5870G + 0.1140B). The analysis explores the
    transformation of 3D color data into a 2D intensity map.

    Args:
        path (str): File path to the input RGB image.

    Note:
        - Output directory: 'experiments/activity_001/results/color_change'.
    """
    logger.info("Start image color change only band transformation.")

    path_save = "experiments/activity_001/results/color_change"

    img = upload_images(path=path, color=True)
    views.view_simplified(
        image=img,
        title="Imagem Original",
        save=True,
        path_save=path_save,
        name_save="original_image",
        plot=False,
    )

    transformation = color_change_only_band(imagen=img)

    save_images(
        image=transformation,
        path=path_save,
        name_save="color_change_onlyband_raw.png",
    )

    views.view_comparison(
        img_orig=img,
        img_trand=transformation,
        info_trasd="Transformação de cores",
        path_save=path_save,
        name_save="color_change_onlyband_comp",
        save=True,
        bar=False,
        plot=False,
    )

    del transformation, img
    logger.info("Completion of color change only band transformation")


# bitplanes
def bitplanes_analysis(path: str):
    """Executes a systematic decomposition and visual analysis of image bit-planes.

    This function loads a grayscale image and decomposes it into its 8
    constituent bit-planes (from bit 0 to bit 7). It automates the storage
    of each individual binary mask and generates composite visualizations
    to compare the informational value of Least Significant Bits (LSB)
    versus Most Significant Bits (MSB).

    Args:
        path (str): File path to the source image (grayscale) to be analyzed.

    Note:
        - Results are organized in 'experiments/activity_001/results/bitplanes'.
        - Saves 8 raw binary images named 'bitplanes_{0-7}_raw.png'.
    """
    logger.info("Start image bitplanes transformation.")

    path_save = "experiments/activity_001/results/bitplanes"

    img = upload_images(path=path, color=False)
    views.view_simplified(
        image=img,
        title="Imagem Original",
        save=True,
        path_save=path_save,
        name_save="original_image",
        plot=False,
    )

    transformations = bitplanes(imagen=img)

    for bit, transformation in enumerate(transformations):
        save_images(
            image=transformation,
            path=path_save,
            name_save=f"bitplanes_{bit}_raw.png",
        )

    views.view_comparison_mult(
        img_orig=img,
        img_trand_1=transformations[0],
        img_trand_2=transformations[4],
        info_trasd_1=f"Plano de bits {0}",
        info_trasd_2=f"Plano de bits {4}",
        path_save=path_save,
        name_save="bitplanes_04_comp",
        save=True,
        bar=False,
        plot=False,
    )
    views.view_comparison_mult(
        img_orig=img,
        img_trand_1=transformations[4],
        img_trand_2=transformations[7],
        info_trasd_1=f"Plano de bits {4}",
        info_trasd_2=f"Plano de bits {7}",
        path_save=path_save,
        name_save="bitplanes_47_comp",
        save=True,
        bar=False,
        plot=False,
    )

    del transformations, img
    logger.info("Completion of bitplanes transformation")


# combination images
def combination_images_analysis(path_a: str, path_b: str):
    """Executes a systematic analysis of image blending through linear combination.

    This function loads two grayscale images and merges them using a set of
    predefined weight pairs [alpha_a, alpha_b]. It automates the generation
    of raw blended images and creates triple-panel visual comparisons to
    evaluate the influence of each source image on the final composite.

    Args:
        path_a (str): File path to the first source image (Image A).
        path_b (str): File path to the second source image (Image B).

    Note:
        - Results are stored in 'experiments/activity_001/results/combination_images'.
        - Tested blending factors [alpha_a, alpha_b]:
            1. [0.2, 0.8]: High dominance of Image B.
            2. [0.5, 0.5]: Equal contribution from both images.
            3. [0.8, 0.2]: High dominance of Image A.
    """

    logger.info("Start image combination images transformation.")

    path_save = "experiments/activity_001/results/combination_images"

    img_a = upload_images(path=path_a, color=False)
    img_b = upload_images(path=path_b, color=False)

    for factor in [[0.2, 0.8], [0.5, 0.5], [0.8, 0.2]]:
        transformation = combination_images(
            imagen_a=img_a, imagen_b=img_b, factor=factor
        )

        save_images(
            image=transformation,
            path=path_save,
            name_save=f"combination_images_{factor}_raw.png",
        )

        views.view_comparison_mult(
            img_orig=img_a,
            img_trand_1=img_b,
            img_trand_2=transformation,
            img_trand_orig="Imagem A",
            info_trasd_1="Imagem B",
            info_trasd_2=rf"Combinação ($\alpha_a={factor[0]}; \alpha_b={factor[1]}$)",
            path_save=path_save,
            name_save=f"combination_images_{factor}_comp.png",
            save=True,
            bar=False,
            plot=False,
        )

    del transformation, img_a, img_b
    logger.info("Completion of combination images transformation")

import logging
import cv2
from pathlib import Path

from activity_002.src.thresholding import Thresholding
from activity_002.src.transitions import Transitions
from activity_002.utilities.graphics import Graphics
from activity_002.utilities.load_save import upload_images

logger = logging.getLogger(__name__)

g = Graphics()

paths_imgs = {
    "baboon": "data/baboon.pgm",
    "fiducial": "data/fiducial.pgm",
    "monarch": "data/monarch.pgm",
    "peppers": "data/peppers.pgm",
    "retina": "data/retina.pgm",
    "sonnet": "data/sonnet.pgm",
    "wedge": "data/wedge.pgm",
}

thresholds = {
    "global_method": [64, 127, 192],
    "otsu_method": [0.3, 1, 1.3],
    "average_method": [-5.0, 2.0, 15.0],
    "niblack_method": [-0.5, -0.2, 0.2],
    "sauvola_pietaksinen_method": [0.1, 0.3, 0.5],
    "contrast_method": [64, 127, 192],
    "phansalskar_more_sabale_method": [0.1, 0.25, 0.5],
}


def thresholding_analysis(paths_res: dict):
    logger.info("Start thresholding analysis")

    thresholding = Thresholding(paths_res=paths_res)
    for _idx, path_img in paths_imgs.items():
        img = upload_images(path=path_img, color=False)

        for thresh in thresholds["global_method"]:
            logger.info("Thresholding analysis: global_method")
            thresholding.global_method(img=img, thresh=thresh, name_img=_idx)

        logger.info("Thresholding analysis: otsu_method")
        thresholding.otsu_method(img=img, name_img=_idx)

        logger.info("Thresholding analysis: median_method")
        thresholding.median_method(img=img, name_img=_idx)

        for thresh in thresholds["average_method"]:
            logger.info("Thresholding analysis: average_method")
            thresholding.average_method(img=img, thresh=thresh, name_img=_idx)

        for thresh in thresholds["niblack_method"]:
            logger.info("Thresholding analysis: niblack_method")
            thresholding.niblack_method(img=img, thresh=thresh, name_img=_idx)

        for thresh in thresholds["sauvola_pietaksinen_method"]:
            logger.info("Thresholding analysis: sauvola_pietaksinen_method")
            thresholding.sauvola_pietaksinen_method(
                img=img, thresh=thresh, name_img=_idx
            )

        for thresh in thresholds["contrast_method"]:
            logger.info("Thresholding analysis: contrast_method")
            thresholding.contrast_method(img=img, thresh=thresh, name_img=_idx)

        thresholding.bernsen_method(img=img, name_img=_idx)

        for thresh in thresholds["phansalskar_more_sabale_method"]:
            logger.info("Thresholding analysis: phansalskar_more_sabale_method")
            thresholding.phansalskar_more_sabale_method(
                img=img, thresh=thresh, name_img=_idx
            )

    logging.info("Thresholding analysis completion")


videos_paths = {
    "toy": "data/toy.mp4",
    "umn": "data/umn.mp4",
    "xylophone": "data/xylophone.mp4",
}


def transitions_analysis(paths_res: dict):
    transitions = Transitions(paths_res=paths_res)

    transitions.detect_pixels(
        video_path=videos_paths["toy"],
        T1=30,
        T2_factor=0.025,
        name_img="toy",
    )
    transitions.detect_pixels(
        video_path=videos_paths["umn"],
        T1=30,
        T2_factor=0.04,
        name_img="umn",
    )
    transitions.detect_pixels(
        video_path=videos_paths["xylophone"],
        T1=30,
        T2_factor=0.03,
        name_img="xylophone",
    )

    transitions.detect_blocks(
        video_path=videos_paths["toy"],
        T1=30,
        T2_factor=0.25,
        block_size=16,
        name_img="toy",
    )

    transitions.detect_blocks(
        video_path=videos_paths["umn"],
        T1=30,
        T2_factor=0.26,
        block_size=16,
        name_img="umn",
    )

    transitions.detect_blocks(
        video_path=videos_paths["xylophone"],
        T1=30,
        T2_factor=0.32,
        block_size=16,
        name_img="xylophone",
    )

    transitions.detect_histograms(
        video_path=videos_paths["toy"],
        alpha=2.1,
        bins=30,
        name_img="toy",
    )

    transitions.detect_histograms(
        video_path=videos_paths["umn"],
        alpha=2,
        bins=30,
        name_img="umn",
    )

    transitions.detect_histograms(
        video_path=videos_paths["xylophone"],
        alpha=1.38,
        bins=30,
        name_img="xylophone",
    )

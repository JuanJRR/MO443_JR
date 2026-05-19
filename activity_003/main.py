import logging

from utilities.environment_initialization import folder_initialization
from utilities.graphics import Graphics
from utilities.logging_settings import logging_settings

from activity_003.src.trasformation import TrasformationFFT2D

logging_settings(file_level=logging.INFO, console_level=logging.INFO)
logger = logging.getLogger(__name__)


logger.info("Creating the folders to store the results")
paths_report = {
    "camera": "activity_003/results/camera",
    "baboon": "activity_003/results/baboon",
    "fiducial": "activity_003/results/fiducial",
    "monarch": "activity_003/results/monarch",
    "peppers": "activity_003/results/peppers",
    "retina": "activity_003/results/retina",
    "monedas": "activity_003/results/monedas",
}

folder_initialization(paths_folder=paths_report)
logger.info("Creation completed.")

grah = Graphics()

paths_imgs = {
    "camera": "data/camera.png",
    "baboon": "data/baboon.pgm",
    "fiducial": "data/fiducial.pgm",
    "monarch": "data/monarch.pgm",
    "peppers": "data/peppers.pgm",
    "retina": "data/retina.pgm",
    "monedas": "data/monedas.jpg",
}

logger.info("start of the analysis")
for idx, value in paths_imgs.items():

    analysis_fft = TrasformationFFT2D(path_img=value)
    results = analysis_fft.comparative_analysis()

    grah.view_multiple_analysis(
        results=results,
        path_save=paths_report[idx],
        name_save=idx,
        save=True,
        plot=False,
    )

logger.info("end of analysis")
import logging

from utilities.environment_initialization import folder_initialization
from utilities.logging_settings import logging_settings

from activity_002.src.analysis import thresholding_analysis
from activity_002.src.analysis import transitions_analysis

logging_settings(file_level=logging.INFO, console_level=logging.INFO)
logger = logging.getLogger(__name__)

logger.info("Creating the folders to store the results")
paths_report_thresholding = {
    "average_method": "activity_002/results/thresholding/average_method",
    "bernsen_method": "activity_002/results/thresholding/bernsen_method",
    "contrast_method": "activity_002/results/thresholding/contrast_method",
    "global_method": "activity_002/results/thresholding/global_method",
    "median_method": "activity_002/results/thresholding/median_method",
    "niblack_method": "activity_002/results/thresholding/niblack_method",
    "otsu_method": "activity_002/results/thresholding/otsu_method",
    "phansalskar_more_sabale_method": "activity_002/results/thresholding/phansalskar_more_sabale_method",
    "sauvola_pietaksinen_method": "activity_002/results/thresholding/sauvola_pietaksinen_method",
}

paths_report_transitions = {
    "differences_between_blocks": "activity_002/results/detection_transitions_videos/differences_between_blocks",
    "differences_between_histograms": "activity_002/results/detection_transitions_videos/differences_between_histograms",
    "differences_between_pixels": "activity_002/results/detection_transitions_videos/differences_between_pixels",
}


folder_initialization(paths_folder=paths_report_thresholding)
folder_initialization(paths_folder=paths_report_transitions)
logger.info("Creation completed.")

thresholding_analysis(paths_res=paths_report_thresholding)

transitions_analysis(paths_res=paths_report_transitions)

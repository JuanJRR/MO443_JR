import logging

from src.activity_001.analysis import (
    binarization_threshold_analysis,
    brightness_adjustment_analysis,
    mosaics_analysis,
    pencil_sketch_analysis,
    rotation_analysis,
    scaling_analysis,
    color_change_filter_analysis,
    color_change_operation_analysis,
    color_change_onlyband_analysis,
    bitplanes_analysis,
    combination_images_analysis,
)
from src.utilities.logging_settings import logging_settings

logging_settings(file_level=logging.INFO)
logger = logging.getLogger(__name__)

image_paths = {
    "baboon": "/app/data/baboon_monocromatica.png",
    "watch": "/app/data/watch.png",
    "city": "/app/data/city.png",
    "butterfly": "data/butterfly.png",
}

# rotation_analysis(path=image_paths["baboon"])
# scaling_analysis(path=image_paths["baboon"])
# pencil_sketch_analysis(path=image_paths["watch"])
# brightness_adjustment_analysis(path=image_paths["baboon"])
# binarization_threshold_analysis(path=image_paths["baboon"])
# mosaics_analysis(path=image_paths["baboon"])
# color_change_filter_analysis(path=image_paths["watch"])
# color_change_operation_analysis(path=image_paths["watch"])
# color_change_onlyband_analysis(path=image_paths["watch"])
# bitplanes_analysis(path=image_paths["baboon"])
combination_images_analysis(
    path_a=image_paths["baboon"], path_b=image_paths["butterfly"]
)

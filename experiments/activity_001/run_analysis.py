import logging

from src.activity_001.analysis import rotation_analysis, scaling_analysis
from src.utilities.logging_settings import logging_settings

logging_settings(file_level=logging.INFO)
logger = logging.getLogger(__name__)

image_paths = {
    "baboon": "/app/data/baboon_monocromatica.png",
    "watch": "/app/data/watch.png",
    "city": "/app/data/city.png",
}

# rotation_analysis(path=image_paths["baboon"])
scaling_analysis(path=image_paths["baboon"])

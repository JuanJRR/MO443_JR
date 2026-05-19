import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def folder_initialization(paths_folder: dict):
    for _idx, path_report in paths_folder.items():
        folder = Path(path_report).resolve()
        try:
            if not os.path.exists(folder):
                folder.mkdir(parents=True, exist_ok=True)
                logging.info(f"Folder created successfully: {folder}")
            else:
                logging.info(f"The folder {folder} already exists.")
        except PermissionError:
            logging.error(f"Insufficient permissions to create: {folder}.")
        except OSError as e:
            logging.error(f"The folder {folder} could not be created. Reason: {e}")
        except Exception as e:
            logging.error(f"An unexpected error occurred while handling {folder}: {e}")

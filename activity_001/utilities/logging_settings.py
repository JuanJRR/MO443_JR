import logging
import sys


def logging_settings(console_level=logging.INFO, file_level=logging.DEBUG):
    """
    Configures the global logging system for the project.

    This utility sets up a dual-handler logging architecture: one for real-time 
    console output and another for persistent file storage. It is pre-configured 
    to work within the Docker container environment.

    Args:
        console_level (_type_, optional): Minimum logging level for the console 
            output. Defaults to logging.INFO. Defaults to logging.INFO.
        file_level (_type_, optional): Minimum logging level for the log file. Defaults to logging.DEBUG.

    Note:
        - The log file is saved at '/app/logger.log' within the container.
        - The system ensures handlers are only added if they do not already 
          exist to avoid duplicate log entries.
    """
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # Desired format for messages
    format = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # CONSOLE HANDLER (StreamHandler)
    if not logger.handlers:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(console_level)
        console_handler.setFormatter(format)
        logger.addHandler(console_handler)

        # FILE HANDLER - Optional but recommended
        file_handler = logging.FileHandler("/app/logger.log", mode="w")
        file_handler.setLevel(file_level)
        file_handler.setFormatter(format)
        logger.addHandler(file_handler)

        # 4. Mensaje de inicio
        logger.info("Logging System Initialized.")
        logger.info(
            f"Console Level: {logging.getLevelName(console_level)}, File Level: {logging.getLevelName(file_level)}"
        )
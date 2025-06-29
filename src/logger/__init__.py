import logging
import sys, os
from datetime import datetime

from logging.handlers import RotatingFileHandler

# constants for log configuration
LOG_DIR = "logs"
LOG_FILE = f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log"
MAX_LOG_SIZE = 5 * 1024 * 1024  # 5 MB
BACKUP_COUNT = 3

# Construct log file path
# dynamically get the root directory of the project
root_dir = os.path.dirname(os.path.abspath(os.path.join(os.path.dirname(__file__), '../')))
log_dir_path = os.path.join(root_dir, LOG_DIR)
# Ensure the log directory exists
os.makedirs(log_dir_path, exist_ok=True)
log_file_path = os.path.join(log_dir_path, LOG_FILE)

# Configure logging
def configure_logging():
    """Configure logging for the application."""
    # create a custom logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # define formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # create a rotating file handler
    file_handler = RotatingFileHandler(log_file_path, maxBytes=MAX_LOG_SIZE, backupCount=BACKUP_COUNT,encoding='utf-8')

    # set the formatter for the file handler
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # add the file handler to the logger
    logger.addHandler(file_handler)

    # console handler for output to console
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    logger.addHandler(console_handler)

configure_logging()
print(root_dir)

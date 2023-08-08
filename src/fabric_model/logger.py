import logging
#from typing import Logger

def get_logger(path) -> logging.Logger:
    # Configure log format
    log_format = "%(asctime)s %(module)15s %(levelname)8s : %(message)s"
    date_format = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(format=log_format, datefmt=date_format, level=logging.DEBUG)

    # Define log file handler
    log_file = path
    file_handler = logging.FileHandler(log_file)

    # Configure log file handler
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)

    # Add the log file handler to the root logger
    logging.getLogger(__name__).addHandler(file_handler)
    return logging.getLogger(__name__)
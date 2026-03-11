# src/exp2_lendingclub/model_training/utils_logger.py

import logging
import os
from .config import LOG_DIR

def get_logger(name: str, log_file: str):
    """
    Create a logger that logs to both console and a file.
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if called multiple times
    if logger.handlers:
        return logger

    log_path = os.path.join(LOG_DIR, log_file)
    fh = logging.FileHandler(log_path)
    ch = logging.StreamHandler()

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)
    ch.setFormatter(fmt)

    logger.addHandler(fh)
    logger.addHandler(ch)

    return logger

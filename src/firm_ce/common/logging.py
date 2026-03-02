import logging
import os
from datetime import datetime

logging.getLogger("numba").setLevel(logging.WARNING)


def get_logger() -> logging.Logger:
    """
    Return the firm_ce logger.

    Returns:
    -------
    logging.Logger: The firm_ce logger instance.
    """
    return logging.getLogger("firm_ce")


def init_model_logger(model_name: str, logging_flag: bool) -> str:
    """
    Initialize a logger for the model run, configured to log both to console and a log file.
    Logger does not work within JIT compiled code.

    The logger writes:
    - All messages (DEBUG and above) to a log file in a new `results/Model_<timestamp>` directory.
    - INFO and higher messages to the console.

    Use get_logger() to retrieve the configured logger instance from anywhere in the codebase.

    Parameters:
    -------
    model_name (str): Name used to label the results directory, based on the model instance name.
    logging_flag (bool): If True, writes to a timestamped directory. If False, writes to
        `results/temp`.

    Returns:
    -------
    str: Path to the results directory created for this model run.
    """

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if logging_flag:
        results_dir = os.path.join("results", f"{model_name}_{timestamp}")
    else:
        results_dir = os.path.join("results", "temp")
    os.makedirs(results_dir, exist_ok=True)

    log_path = os.path.join(results_dir, "log.txt")

    logger = logging.getLogger("firm_ce")
    logger.setLevel(logging.DEBUG)
    logger.handlers.clear()
    logger.propagate = False

    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    file_handler.setFormatter(file_format)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter("%(message)s"))

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

    logger.info(f"Logger initialized. Writing to {log_path}")
    return results_dir

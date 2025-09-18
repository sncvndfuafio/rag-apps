import logging

def get_logger(name: str):
    """Configures and returns a logger instance."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    # Create console handler and set level to info
    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)

    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    # Add formatter to ch
    ch.setFormatter(formatter)

    # Add ch to logger
    if not logger.handlers:
        logger.addHandler(ch)

    return logger
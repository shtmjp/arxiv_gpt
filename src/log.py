from logging import Formatter, Logger, StreamHandler, getLogger


def get_logger_for_test(name: str) -> Logger:
    logger = getLogger(name)
    handler = StreamHandler()
    handler.setFormatter(Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
    logger.addHandler(handler)
    logger.setLevel("DEBUG")
    return logger

import logging
from .paths import LOGS_DIR, TRAIN_LOG_PATH

logger = logging.getLogger("hybridtrinet")

if not logger.handlers:
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(TRAIN_LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.propagate = False  # không đẩy lên root logger

    logger.info("Logger initialized, writing to %s", TRAIN_LOG_PATH)

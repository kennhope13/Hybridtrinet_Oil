import logging
from .paths import LOGS_DIR, TRAIN_LOG_PATH

# Tạo logger dùng chung
logger = logging.getLogger("hybridtrinet")

# Tránh add handler nhiều lần khi Streamlit reload
if not logger.handlers:
    logger.setLevel(logging.INFO)

    # Ghi ra file train.log
    fh = logging.FileHandler(TRAIN_LOG_PATH, encoding="utf-8")
    fh.setLevel(logging.INFO)

    fmt = logging.Formatter(
        "%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    fh.setFormatter(fmt)

    logger.addHandler(fh)
    logger.propagate = False  # không đẩy lên root logger

    # Dòng test: chỉ cần import logger là file log phải có dòng này
    logger.info("Logger initialized, writing to %s", TRAIN_LOG_PATH)

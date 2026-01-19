# forecast_app/config.py
import os
from functools import lru_cache
from pathlib import Path

from src.utils.paths import BASE_DIR, RUN_OUTPUT_DIR
from src.utils.config_loader import load_yaml_config, load_env_secrets


@lru_cache(maxsize=1)
def get_cfg():
    cfg = load_yaml_config()
    load_env_secrets()
    return cfg


def get_defaults():
    cfg = get_cfg()

    date_col = cfg.get("default_date_col", "Ngày")
    default_h_next = int(cfg.get("default_h_next", 5))

    default_clean_rel = cfg.get(
        "default_clean_path", "data/base/du_lieu_noi_suy_clean_updated_end_14-11.xlsx"
    )
    default_clean_path = (BASE_DIR / default_clean_rel).resolve()

    fred_api_key_default = os.getenv("FRED_API_KEY", "")

    return {
        "DATE_COL_CFG": date_col,
        "DEFAULT_H_NEXT": default_h_next,
        "DEFAULT_CLEAN_PATH": default_clean_path,
        "FRED_API_KEY_DEFAULT": fred_api_key_default,
        "BASE_DIR": BASE_DIR,
        "RUN_OUTPUT_DIR": RUN_OUTPUT_DIR,
    }


# constants
TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
K = 64
H = 14
VAL_RATIO = 0.10
SEED = 42

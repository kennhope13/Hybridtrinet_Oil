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
    train = cfg.get("default_train", {}) or {}

    default_epochs = int(train.get("epochs", 300))
    default_lr = float(train.get("lr", 3e-4))
    default_batch_gpu = int(train.get("batch_gpu", 128))
    default_batch_cpu = int(train.get("batch_cpu", 64))
    default_ensemble_n = int(train.get("ensemble_n", 3))

    default_focus_h = int(train.get("focus_h", 5))
    default_focus_w = float(train.get("focus_w", 3.5))

    default_alpha_delta = float(train.get("alpha_delta", 0.2))
    default_beta_price  = float(train.get("beta_price", 1.0))
    default_eps_mape    = float(train.get("eps_mape", 1e-3))

    default_loss = str(train.get("loss", "huber"))
    default_patience = int(train.get("patience", 25))
    default_wd = float(train.get("wd", 1e-4))
    default_clip = float(train.get("clip", 1.0))
    default_amp = bool(train.get("amp", True))
    return {
        "DATE_COL_CFG": date_col,
        "DEFAULT_H_NEXT": default_h_next,
        "DEFAULT_CLEAN_PATH": default_clean_path,
        "FRED_API_KEY_DEFAULT": fred_api_key_default,
        "BASE_DIR": BASE_DIR,
        "RUN_OUTPUT_DIR": RUN_OUTPUT_DIR,
        "DEFAULT_EPOCHS": default_epochs,
        "DEFAULT_LR": default_lr,
        "DEFAULT_BATCH_GPU": default_batch_gpu,
        "DEFAULT_BATCH_CPU": default_batch_cpu,
        "DEFAULT_ENSEMBLE_N": default_ensemble_n,
        "DEFAULT_FOCUS_H": default_focus_h,
        "DEFAULT_FOCUS_W": default_focus_w,
        "DEFAULT_ALPHA_DELTA": default_alpha_delta,
        "DEFAULT_BETA_PRICE": default_beta_price,
        "DEFAULT_EPS_MAPE": default_eps_mape,
        "DEFAULT_LOSS": default_loss,
        "DEFAULT_PATIENCE": default_patience,
        "DEFAULT_WD": default_wd,
        "DEFAULT_CLIP": default_clip,
        "DEFAULT_AMP": default_amp,
    }


# constants
TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
K = 64
H = 5
VAL_RATIO = 0.10
SEED = 42

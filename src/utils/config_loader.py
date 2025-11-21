# src/utils/config_loader.py

import os
from pathlib import Path

import yaml

from .paths import BASE_DIR

CONFIG_DIR = BASE_DIR / "config"
YAML_CONFIG_PATH = CONFIG_DIR / "streamlit_config.yaml"
SECRETS_ENV_PATH = CONFIG_DIR / "secrets.env"


def load_yaml_config() -> dict:
    """Đọc file streamlit_config.yaml, trả về dict (có thể rỗng)."""
    if not YAML_CONFIG_PATH.exists():
        return {}
    with open(YAML_CONFIG_PATH, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data or {}


def load_env_secrets():
    """Đọc secrets.env và set vào os.environ nếu chưa có."""
    if not SECRETS_ENV_PATH.exists():
        return
    text = SECRETS_ENV_PATH.read_text(encoding="utf-8")
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        k, v = line.split("=", 1)
        k = k.strip()
        v = v.strip()
        # Không overwrite nếu env đã có (cho phép override từ ngoài)
        if k and (k not in os.environ):
            os.environ[k] = v

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
RUN_OUTPUT_DIR = ARTIFACTS_DIR / "run_output"
CONFIG_DIR = BASE_DIR / "config"

LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(parents=True, exist_ok=True)  # đảm bảo tồn tại

TRAIN_LOG_PATH = LOGS_DIR / "train.log"
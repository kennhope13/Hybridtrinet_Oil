# app_forecast.py (ENTRY)
import sys
from pathlib import Path
import torch

# === Find project root (robust): pick the nearest parent containing "src" ===
THIS = Path(__file__).resolve()
candidates = [THIS.parent, THIS.parent.parent, THIS.parent.parent.parent]
ROOT = next((p for p in candidates if (p / "src").exists()), THIS.parent)

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# streamlit + torch classes workaround
torch.classes.__path__ = []

from forecast_app.core import main

if __name__ == "__main__":
    main()

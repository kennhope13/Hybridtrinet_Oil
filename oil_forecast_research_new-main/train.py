from pathlib import Path
import random

import sys
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

# Fix encoding for terminal
if sys.stdout.encoding != 'utf-8':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from src.model.dataset import DataProcessor, PetroleumDataset
from src.model.model import GUMNet


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_data_exo_ver1.csv"
CKPT_DIR = BASE_DIR / "checkpoints"
CKPT_PATH = CKPT_DIR / "gumnet_ckpt.pt"

DATE_COL = "Ngày"
TARGET_COLS = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]
   # đổi nếu muốn train nhiều cột
SEQ_LEN = 30
HORIZON = 5
NUM_QUANTILES = 3
QUANTILES = [0.1, 0.5, 0.9]

BATCH_SIZE = 32
EPOCHS = 150
LR = 1e-3
VAL_RATIO = 0.2
SEED = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def read_data(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    if DATE_COL not in df.columns:
        raise ValueError(f"Column {DATE_COL} not found")

    df[DATE_COL] = pd.to_datetime(df[DATE_COL], errors="coerce")
    df = df.dropna(subset=[DATE_COL]).sort_values(DATE_COL).reset_index(drop=True)

    for c in df.columns:
        if c != DATE_COL:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    numeric_cols = [c for c in df.columns if c != DATE_COL]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear").bfill().ffill()

    return df


def quantile_loss(pred, target, quantiles):
    # pred: [B, H, O, Q]
    # target: [B, H, O]
    loss = 0.0
    for i, q in enumerate(quantiles):
        err = target - pred[..., i]
        loss_q = torch.maximum((q - 1) * err, q * err)
        loss = loss + loss_q.mean()
    return loss / len(quantiles)


def main():
    set_seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    df = read_data(DATA_PATH)
    feature_cols = [c for c in df.columns if c != DATE_COL]

    for c in TARGET_COLS:
        if c not in df.columns:
            raise ValueError(f"Missing target column: {c}")

    processor = DataProcessor(seq_len=SEQ_LEN, horizon=HORIZON)
    X, y = processor.prepare_data(
        df=df,
        target_cols=TARGET_COLS,
        feature_cols=feature_cols,
        is_train=True,
    )

    if len(X) == 0:
        raise ValueError("Not enough data to create training sequences.")

    split_idx = int(len(X) * (1 - VAL_RATIO))
    X_train, X_val = X[:split_idx], X[split_idx:]
    y_train, y_val = y[:split_idx], y[split_idx:]

    train_loader = DataLoader(PetroleumDataset(X_train, y_train), batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(PetroleumDataset(X_val, y_val), batch_size=BATCH_SIZE, shuffle=False)

    model = GUMNet(
        seq_len=SEQ_LEN,
        input_dim=len(feature_cols),
        output_dim=len(TARGET_COLS),
        horizon=HORIZON,
        num_quantiles=NUM_QUANTILES,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val = float("inf")
    CKPT_DIR.mkdir(parents=True, exist_ok=True)

    for epoch in range(EPOCHS):
        model.train()
        train_losses = []

        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad()
            pred, _ = model(xb)
            loss = quantile_loss(pred, yb, QUANTILES)
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device)
                yb = yb.to(device)

                pred, _ = model(xb)
                loss = quantile_loss(pred, yb, QUANTILES)
                val_losses.append(loss.item())

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        print(f"Epoch {epoch+1}/{EPOCHS} - train_loss={train_loss:.6f} - val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "seq_len": SEQ_LEN,
                    "horizon": HORIZON,
                    "num_quantiles": NUM_QUANTILES,
                    "quantiles": QUANTILES,
                    "feature_cols": feature_cols,
                    "target_cols": TARGET_COLS,
                    "input_dim": len(feature_cols),
                    "output_dim": len(TARGET_COLS),
                    "feature_scaler": processor.feature_scaler,
                    "target_scaler": processor.target_scaler,
                    "date_col": DATE_COL,
                },
                CKPT_PATH,
            )
            print(f"Saved best checkpoint: {CKPT_PATH}")

    print("Training finished.")


if __name__ == "__main__":
    main()
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.model.model import GUMNet


BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "processed" / "clean_data_exo_ver1.csv"
CKPT_PATH = BASE_DIR / "checkpoints" / "gumnet_ckpt.pt"
OUT_PATH = BASE_DIR / "results" / "forecast.csv"


def read_data(path: Path, date_col: str) -> pd.DataFrame:
    if path.suffix.lower() in [".xlsx", ".xls"]:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)

    for c in df.columns:
        if c != date_col:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    numeric_cols = [c for c in df.columns if c != date_col]
    df[numeric_cols] = df[numeric_cols].interpolate(method="linear").bfill().ffill()
    return df


def next_business_days(last_date: pd.Timestamp, n: int):
    days = []
    d = pd.Timestamp(last_date)
    while len(days) < n:
        d += pd.Timedelta(days=1)
        if d.weekday() < 5:
            days.append(d)
    return days


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ckpt = torch.load(CKPT_PATH, map_location=device, weights_only=False)

    date_col = ckpt["date_col"]
    feature_cols = ckpt["feature_cols"]
    target_cols = ckpt["target_cols"]

    df = read_data(DATA_PATH, date_col)

    model = GUMNet(
        seq_len=ckpt["seq_len"],
        input_dim=ckpt["input_dim"],
        output_dim=ckpt["output_dim"],
        horizon=ckpt["horizon"],
        num_quantiles=ckpt["num_quantiles"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    feature_scaler = ckpt["feature_scaler"]
    target_scaler = ckpt["target_scaler"]

    X_all = feature_scaler.transform(df[feature_cols].values)
    x_last = X_all[-ckpt["seq_len"]:]
    x_last = torch.tensor(x_last, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred_scaled, weights = model(x_last)

    # lấy quantile giữa (median)
    q_idx = ckpt["num_quantiles"] // 2
    pred_scaled = pred_scaled.cpu().numpy()[0, :, :, q_idx]   # [H, O]
    pred = target_scaler.inverse_transform(pred_scaled)

    future_dates = next_business_days(pd.to_datetime(df[date_col].iloc[-1]), ckpt["horizon"])

    out_df = pd.DataFrame(pred, columns=[f"{c}_pred" for c in target_cols])
    out_df.insert(0, date_col, future_dates)

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(OUT_PATH, index=False, encoding="utf-8-sig")

    print("Dự đoán xong:")
    print(out_df)
    print("Gate weights:", weights.cpu().numpy())
    print(f"Đã lưu: {OUT_PATH}")


if __name__ == "__main__":
    main()
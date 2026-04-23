from pathlib import Path
import pandas as pd

from src.metrics import mae, rmse, mape


BASE_DIR = Path(__file__).resolve().parent
PRED_PATH = BASE_DIR / "results" / "forecast.csv"
ACTUAL_PATH = BASE_DIR / "data" / "processed" / "actual_future.csv"
DATE_COL = "Ngày"
TARGET_COLS = ["MG95"]   # sửa nếu cần


def read_table(path: Path):
    if not path.exists():
        return None
    if path.suffix.lower() in [".xlsx", ".xls"]:
        return pd.read_excel(path)
    return pd.read_csv(path)


def main():
    pred = read_table(PRED_PATH)
    if pred is None:
        raise FileNotFoundError(f"Không tìm thấy file forecast: {PRED_PATH}")

    actual = read_table(ACTUAL_PATH)
    if actual is None:
        print(f"Chưa có file dữ liệu thực tế để evaluate: {ACTUAL_PATH}")
        print("Hãy tạo file actual_future.csv hoặc actual_future.xlsx rồi chạy lại.")
        return

    pred[DATE_COL] = pd.to_datetime(pred[DATE_COL], errors="coerce")
    actual[DATE_COL] = pd.to_datetime(actual[DATE_COL], errors="coerce")

    df = pred.merge(actual, on=DATE_COL, how="inner")

    if df.empty:
        print("Không có ngày giao nhau giữa forecast và actual.")
        return

    rows = []
    for c in TARGET_COLS:
        pred_col = f"{c}_pred"
        real_col = c

        if pred_col not in df.columns or real_col not in df.columns:
            print(f"Bỏ qua {c} vì thiếu cột {pred_col} hoặc {real_col}")
            continue

        y_pred = df[pred_col].values
        y_true = df[real_col].values

        rows.append({
            "target": c,
            "MAE": mae(y_true, y_pred),
            "RMSE": rmse(y_true, y_pred),
            "MAPE": mape(y_true, y_pred),
        })

    if not rows:
        print("Không tính được metric nào.")
        return

    result = pd.DataFrame(rows)
    print(result)


if __name__ == "__main__":
    main()
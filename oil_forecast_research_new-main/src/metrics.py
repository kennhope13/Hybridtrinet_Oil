import numpy as np


def mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean(np.abs(y_true - y_pred)))


def mse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def rmse(y_true, y_pred):
    return float(np.sqrt(mse(y_true, y_pred)))


def mape(y_true, y_pred, eps=1e-8):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if y_true.shape != y_pred.shape or y_true.size == 0:
        raise ValueError("MAPE requires nonempty arrays with matching shapes")
    if not np.isfinite(y_true).all() or not np.isfinite(y_pred).all():
        raise ValueError("MAPE requires finite values")
    if eps <= 0 or np.any(np.abs(y_true) <= eps):
        raise ValueError("MAPE is undefined for zero or near-zero actual values")
    return float(np.mean(np.abs((y_true - y_pred) / y_true)) * 100.0)

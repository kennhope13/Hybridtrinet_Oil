from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import copy
import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset


# =========================================================
# Helpers
# =========================================================

def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def infer_date_col(df: pd.DataFrame) -> Optional[str]:
    candidates = ["Ngày", "ngay", "date", "Date", "DATE", "timestamp", "Timestamp"]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def infer_feature_cols(
    df: pd.DataFrame,
    target_cols: List[str],
    date_col: Optional[str] = None,
) -> List[str]:
    if date_col is None:
        date_col = infer_date_col(df)

    exclude = set(target_cols)
    if date_col is not None:
        exclude.add(date_col)

    num_cols = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            num_cols.append(c)

    # luôn đưa target vào feature nếu chưa có
    out = []
    for c in target_cols:
        if c in df.columns and c not in out:
            out.append(c)
    for c in num_cols:
        if c not in out:
            out.append(c)

    return out


def normalize_df(
    df: pd.DataFrame,
    feature_cols: List[str],
    mean_: Optional[pd.Series] = None,
    std_: Optional[pd.Series] = None,
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    x = df[feature_cols].copy()

    if mean_ is None:
        mean_ = x.mean(axis=0)

    if std_ is None:
        std_ = x.std(axis=0).replace(0, 1.0).fillna(1.0)

    x_std = (x - mean_) / std_
    x_std = x_std.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    return x_std, mean_, std_


def split_train_val(
    df: pd.DataFrame,
    val_ratio: float = 0.1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    n = len(df)
    if n < 20:
        return df.copy(), df.copy()

    n_val = max(1, int(round(n * val_ratio)))
    n_val = min(n_val, max(1, n - 1))

    train_df = df.iloc[:-n_val].copy()
    val_df = df.iloc[-n_val:].copy()
    return train_df, val_df


def build_xy_from_std(
    df_std: pd.DataFrame,
    feature_cols: List[str],
    target_cols: List[str],
    k: int,
    h: int,
) -> Tuple[np.ndarray, np.ndarray]:
    arr = df_std[feature_cols].to_numpy(dtype=np.float32)
    tgt_idx = [feature_cols.index(c) for c in target_cols]

    xs = []
    ys = []

    n = len(arr)
    for end_ix in range(k, n - h + 1):
        x = arr[end_ix - k:end_ix, :]                      # [K, D]
        y = arr[end_ix:end_ix + h, :][:, tgt_idx]         # [H, T]
        xs.append(x)
        ys.append(y)

    if not xs:
        return (
            np.zeros((0, k, len(feature_cols)), dtype=np.float32),
            np.zeros((0, h, len(target_cols)), dtype=np.float32),
        )

    return np.stack(xs), np.stack(ys)


class WindowDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray):
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int):
        return self.x[idx], self.y[idx]


def _reshape_model_out(
    out: torch.Tensor,
    batch_size: int,
    h: int,
    t: int,
) -> torch.Tensor:
    """
    Chuẩn hóa output model về [B, H, T].
    Hỗ trợ:
      - [B, H, T]
      - [B, H*T]
      - [B, T] -> [B, 1, T] rồi repeat nếu H=1
    """
    if isinstance(out, (tuple, list)):
        out = out[0]

    if out.ndim == 3:
        return out

    if out.ndim == 2:
        if out.shape[1] == h * t:
            return out.view(batch_size, h, t)
        if out.shape[1] == t and h == 1:
            return out.unsqueeze(1)

    raise ValueError(
        f"Model output shape không hỗ trợ: {tuple(out.shape)}. "
        f"Kỳ vọng [B,H,T] hoặc [B,H*T]."
    )


def weighted_focus_loss(
    pred: torch.Tensor,
    true: torch.Tensor,
    h: int,
    focus_weight: float = 2.0,
    focus_n: int = 5,
    loss_name: str = "smooth_l1",
) -> torch.Tensor:
    """
    pred, true: [B, H, T]
    """
    if loss_name == "mse":
        base = (pred - true) ** 2
    else:
        base = torch.nn.functional.smooth_l1_loss(pred, true, reduction="none")

    w = torch.ones(h, device=pred.device, dtype=pred.dtype)
    w[: min(focus_n, h)] = float(focus_weight)
    w = w.view(1, h, 1)

    return (base * w).mean()


@dataclass
class FineTuneResult:
    model: nn.Module
    feature_cols: List[str]
    target_cols: List[str]
    mean_: pd.Series
    std_: pd.Series
    best_val_loss: float
    history: List[Dict[str, float]]


# =========================================================
# Fine-tune main
# =========================================================

def fine_tune_model(
    model: nn.Module,
    df: pd.DataFrame,
    target_cols: List[str],
    feature_cols: Optional[List[str]] = None,
    date_col: Optional[str] = None,
    k: int = 128,
    h: int = 5,
    val_ratio: float = 0.1,
    epochs: int = 5,
    batch_size: int = 64,
    lr: float = 1e-5,
    weight_decay: float = 1e-4,
    focus_weight: float = 2.0,
    focus_n: int = 5,
    loss_name: str = "smooth_l1",
    shuffle: bool = True,
    seed: Optional[int] = None,
    device: Optional[str] = None,
) -> FineTuneResult:
    """
    Fine-tune model hiện tại trên dữ liệu mới nhất.

    Ý tưởng:
    - Chuẩn hóa feature theo train split
    - Build window [K -> H]
    - Fine-tune vài epoch
    - Trả về model + scaler để forecast tiếp
    """
    if seed is not None:
        set_seed(seed)

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)

    df = df.copy()
    if date_col is None:
        date_col = infer_date_col(df)

    if date_col is not None and date_col in df.columns:
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        df = df.sort_values(date_col).reset_index(drop=True)

    # chỉ giữ các cột cần thiết và drop dòng lỗi target
    existing_targets = [c for c in target_cols if c in df.columns]
    if len(existing_targets) != len(target_cols):
        missing = [c for c in target_cols if c not in df.columns]
        raise ValueError(f"Thiếu target_cols trong df: {missing}")

    df = df.dropna(subset=target_cols).reset_index(drop=True)

    if feature_cols is None:
        feature_cols = infer_feature_cols(df, target_cols=target_cols, date_col=date_col)

    missing_feat = [c for c in feature_cols if c not in df.columns]
    if missing_feat:
        raise ValueError(f"Thiếu feature_cols trong df: {missing_feat}")

    # đủ dữ liệu mới build được window
    if len(df) < (k + h + 5):
        raise ValueError(
            f"Không đủ dữ liệu để fine-tune. "
            f"Cần ít nhất khoảng {k + h + 5} dòng, hiện có {len(df)} dòng."
        )

    train_df, val_df = split_train_val(df, val_ratio=val_ratio)

    train_std, mean_, std_ = normalize_df(train_df, feature_cols=feature_cols)
    val_std, _, _ = normalize_df(val_df, feature_cols=feature_cols, mean_=mean_, std_=std_)

    x_train, y_train = build_xy_from_std(
        df_std=train_std,
        feature_cols=feature_cols,
        target_cols=target_cols,
        k=k,
        h=h,
    )
    x_val, y_val = build_xy_from_std(
        df_std=val_std,
        feature_cols=feature_cols,
        target_cols=target_cols,
        k=k,
        h=h,
    )

    if len(x_train) == 0:
        raise ValueError("Không tạo được sample train để fine-tune.")
    if len(x_val) == 0:
        # nếu val quá ngắn, dùng train cuối làm val fallback
        x_val = x_train[-min(len(x_train), 32):]
        y_val = y_train[-min(len(y_train), 32):]

    ds_train = WindowDataset(x_train, y_train)
    ds_val = WindowDataset(x_val, y_val)

    dl_train = DataLoader(
        ds_train,
        batch_size=min(batch_size, len(ds_train)),
        shuffle=shuffle,
        drop_last=False,
    )
    dl_val = DataLoader(
        ds_val,
        batch_size=min(batch_size, len(ds_val)),
        shuffle=False,
        drop_last=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    history: List[Dict[str, float]] = []

    t = len(target_cols)

    for ep in range(1, epochs + 1):
        model.train()
        train_losses = []

        for xb, yb in dl_train:
            xb = xb.to(device)
            yb = yb.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(xb)
            out = _reshape_model_out(out, batch_size=xb.size(0), h=h, t=t)

            # nếu model trả D lớn hơn T thì cắt T cột đầu
            if out.shape[-1] > t:
                out = out[..., :t]

            loss = weighted_focus_loss(
                pred=out,
                true=yb,
                h=h,
                focus_weight=focus_weight,
                focus_n=focus_n,
                loss_name=loss_name,
            )
            loss.backward()
            optimizer.step()

            train_losses.append(float(loss.item()))

        model.eval()
        val_losses = []

        with torch.no_grad():
            for xb, yb in dl_val:
                xb = xb.to(device)
                yb = yb.to(device)

                out = model(xb)
                out = _reshape_model_out(out, batch_size=xb.size(0), h=h, t=t)

                if out.shape[-1] > t:
                    out = out[..., :t]

                loss = weighted_focus_loss(
                    pred=out,
                    true=yb,
                    h=h,
                    focus_weight=focus_weight,
                    focus_n=focus_n,
                    loss_name=loss_name,
                )
                val_losses.append(float(loss.item()))

        train_loss = float(np.mean(train_losses)) if train_losses else np.nan
        val_loss = float(np.mean(val_losses)) if val_losses else np.nan

        history.append(
            {
                "epoch": ep,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        if np.isfinite(val_loss) and val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    return FineTuneResult(
        model=model,
        feature_cols=feature_cols,
        target_cols=target_cols,
        mean_=mean_,
        std_=std_,
        best_val_loss=best_val,
        history=history,
    )


# =========================================================
# Checkpoint helpers
# =========================================================

def save_finetuned_bundle(
    save_path: str,
    model: nn.Module,
    feature_cols: List[str],
    target_cols: List[str],
    mean_: pd.Series,
    std_: pd.Series,
    extra: Optional[Dict] = None,
) -> None:
    bundle = {
        "model_state_dict": model.state_dict(),
        "feature_cols": list(feature_cols),
        "target_cols": list(target_cols),
        "mean": mean_.to_dict(),
        "std": std_.to_dict(),
        "extra": extra or {},
    }
    torch.save(bundle, save_path)


def load_bundle_meta(ckpt_path: str) -> Dict:
    obj = torch.load(ckpt_path, map_location="cpu")
    if not isinstance(obj, dict):
        raise ValueError("Checkpoint không đúng định dạng dict.")
    return obj
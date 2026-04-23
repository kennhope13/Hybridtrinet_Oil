# src/model/training.py
from __future__ import annotations

import random
from typing import Tuple

import numpy as np
import torch
from torch.utils.data import Dataset


# =========================
# Repro
# =========================
def set_seed(seed: int = 42):
    seed = int(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# Standardize
# =========================
def standardize(Y: np.ndarray, eps: float = 1e-8):
    """
    Y: (T, D)
    return: Y_std, mu, sd  (mu/sd shape (D,))
    """
    Y = np.asarray(Y, dtype=np.float32)
    if Y.ndim != 2:
        raise ValueError(f"standardize expects (T,D), got {Y.shape}")

    mu = np.nanmean(Y, axis=0).astype(np.float32)
    sd = np.nanstd(Y, axis=0).astype(np.float32)
    sd = np.where(sd < eps, 1.0, sd).astype(np.float32)

    Y_std = (Y - mu) / (sd + eps)
    return Y_std.astype(np.float32), mu.astype(np.float32), sd.astype(np.float32)


# =========================
# Sliding windows
# =========================
def build_windows(Y_std: np.ndarray, K: int, H: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Y_std: (T, D)
    return:
      X: (N, K, D)
      Y: (N, H, D)
    """
    Y_std = np.asarray(Y_std, dtype=np.float32)
    if Y_std.ndim != 2:
        raise ValueError(f"build_windows expects (T,D), got {Y_std.shape}")

    T, D = Y_std.shape
    K = int(K)
    H = int(H)
    N = T - K - H + 1
    if N <= 0:
        raise ValueError(f"Not enough data: T={T}, K={K}, H={H} => N={N}")

    X = np.empty((N, K, D), dtype=np.float32)
    Y = np.empty((N, H, D), dtype=np.float32)
    for i in range(N):
        X[i] = Y_std[i : i + K]
        Y[i] = Y_std[i + K : i + K + H]
    return X, Y


class WindowDS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.Y = torch.as_tensor(Y, dtype=torch.float32)

    def __len__(self):
        return int(self.X.shape[0])

    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]

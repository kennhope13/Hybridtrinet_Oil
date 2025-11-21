import math
import random
from typing import Dict, Any

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from src.utils.logger import logger

USE_L1_LOSS = True
SMOOTHL1_BETA = 0.5
WEIGHT_DECAY = 1e-4
GATE_ENTROPY_LAMBDA = 0.0
MIN_LR = 1e-5
WARMUP_FRAC = 0.10
DEFAULT_SEED = 42


def set_seed(seed: int = DEFAULT_SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def to_tensor(x: np.ndarray) -> torch.Tensor:
    return torch.tensor(x, dtype=torch.float32)


def standardize(arr: np.ndarray, eps: float = 1e-8):
    mu = arr.mean(axis=0, keepdims=True)
    sd = arr.std(axis=0, keepdims=True) + eps
    return (arr - mu) / sd, mu.squeeze(0), sd.squeeze(0)


def build_windows(Y: np.ndarray, k: int, H: int):
    T, D = Y.shape
    N = T - k - H + 1
    if N <= 0:
        raise ValueError("Không đủ dữ liệu cho k, H.")
    X = np.zeros((N, k, D), dtype=np.float32)
    Yh = np.zeros((N, H, D), dtype=np.float32)
    for i in range(N):
        X[i] = Y[i: i + k]
        Yh[i] = Y[i + k: i + k + H]
    return X, Yh


class WindowDS(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray):
        self.X = to_tensor(X)
        self.Y = to_tensor(Y).view(X.shape[0], -1)

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, i):
        return self.X[i], self.Y[i]


class EarlyStopper:
    def __init__(self, patience: int = 80, min_delta: float = 1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best = float("inf")
        self.wait = 0
        self.stop = False

    def step(self, metric: float):
        if metric < self.best - self.min_delta:
            self.best = metric
            self.wait = 0
        else:
            if self.patience is None:
                return
            self.wait += 1
            if self.wait >= self.patience:
                self.stop = True


def _get_criterion(use_l1_loss: bool, smoothl1_beta: float):
    if use_l1_loss:
        return nn.L1Loss()
    return nn.SmoothL1Loss(beta=smoothl1_beta)


def train_one(
    model: torch.nn.Module,
    loader,
    opt: torch.optim.Optimizer,
    device: str = "cuda",
    use_l1_loss: bool = USE_L1_LOSS,
    smoothl1_beta: float = SMOOTHL1_BETA,
    gate_entropy_lambda: float = GATE_ENTROPY_LAMBDA,
) -> float:
    model.train()
    crit = _get_criterion(use_l1_loss, smoothl1_beta)
    tot = 0.0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        opt.zero_grad()
        pred, aux = model(xb)
        loss = crit(pred, yb)

        if gate_entropy_lambda > 0.0:
            _, _, _, w = aux
            ent = -(w * (w.clamp_min(1e-8).log())).sum(dim=-1).mean()
            loss = loss + gate_entropy_lambda * (-ent)

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        opt.step()

        tot += loss.item() * xb.size(0)

    return tot / len(loader.dataset)


@torch.no_grad()
def eval_one(
    model: torch.nn.Module,
    loader,
    H: int,
    D: int,
    device: str = "cuda",
    use_l1_loss: bool = USE_L1_LOSS,
    smoothl1_beta: float = SMOOTHL1_BETA,
):
    model.eval()
    crit = _get_criterion(use_l1_loss, smoothl1_beta)
    tot = 0.0
    per_h = np.zeros(H, dtype=np.float64)
    n_samp = 0

    for xb, yb in loader:
        xb = xb.to(device)
        yb = yb.to(device)

        pred, _ = model(xb)
        loss = crit(pred, yb)
        tot += loss.item() * xb.size(0)

        B = xb.size(0)
        pr = pred.view(B, H, D).cpu().numpy()
        gt = yb.view(B, H, D).cpu().numpy()
        per_h += np.mean(np.abs(pr - gt), axis=(0, 2)) * B
        n_samp += B

    per_h /= max(1, n_samp)
    return tot / len(loader.dataset), per_h


@torch.no_grad()
def eval_metrics_orig(
    model: torch.nn.Module,
    loader,
    H: int,
    D: int,
    mu: np.ndarray,
    sd: np.ndarray,
    device: str = "cuda",
) -> Dict[str, float]:
    model.eval()
    P_list, G_list = [], []

    for xb, yb in loader:
        xb = xb.to(device)
        pr, _ = model(xb)
        pr = pr.view(-1, H, D).cpu().numpy()
        yb = yb.view(-1, H, D).cpu().numpy()
        P_list.append(pr)
        G_list.append(yb)

    P_std = np.vstack(P_list)
    G_std = np.vstack(G_list)

    P = P_std * sd + mu
    G = G_std * sd + mu

    diff = P - G
    abs_diff = np.abs(diff)

    mae = float(abs_diff.mean())
    mse = float((diff ** 2).mean())
    rmse = float(math.sqrt(mse))
    sse = float((diff ** 2).sum())
    sst = float(((G - G.mean()) ** 2).sum())
    r2 = float(1.0 - sse / sst) if sst > 0 else float("nan")

    return {"mae": mae, "mse": mse, "rmse": rmse, "r2": r2}


def fit_model(
    
    model: torch.nn.Module,
    train_ld,
    val_ld,
    H: int,
    D: int,
    epochs: int,
    lr: float,
    mu: np.ndarray,
    sd: np.ndarray,
    device: str = "cuda",
    name: str = "Hybrid",
    status_cb=None,
    use_l1_loss: bool = USE_L1_LOSS,
    smoothl1_beta: float = SMOOTHL1_BETA,
    gate_entropy_lambda: float = GATE_ENTROPY_LAMBDA,
    weight_decay: float = WEIGHT_DECAY,
    min_lr: float = MIN_LR,
    warmup_frac: float = WARMUP_FRAC,
) -> Dict[str, Any]:
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    warmup = max(3, int(warmup_frac * epochs))
    logger.info(
        "Start training %s | device=%s | epochs=%d | lr=%.6f | H=%d | D=%d",
        name, device, epochs, lr, H, D,
    )
    def lr_lambda(ep):
        ep = float(ep)
        if ep < warmup:
            return (ep + 1.0) / warmup
        t = (ep - warmup) / max(1.0, epochs - warmup)
        cos = 0.5 * (1.0 + math.cos(math.pi * t))
        return (min_lr / lr) + (1.0 - min_lr / lr) * cos

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=lr_lambda)
    stopper = EarlyStopper(patience=40, min_delta=5e-4)

    history = {"train": [], "val": [], "mae_orig": [], "lr": []}

    va0, per_h0 = eval_one(
        model,
        val_ld,
        H,
        D,
        device=device,
        use_l1_loss=use_l1_loss,
        smoothl1_beta=smoothl1_beta,
    )

    P_list, G_list = [], []
    with torch.no_grad():
        for xb, yb in val_ld:
            xb = xb.to(device)
            pr, _ = model(xb)
            pr = pr.view(-1, H, D).cpu().numpy()
            yb = yb.view(-1, H, D).cpu().numpy()
            P_list.append(pr)
            G_list.append(yb)

    P0 = np.vstack(P_list) * sd + mu
    G0 = np.vstack(G_list) * sd + mu
    mae0 = float(np.mean(np.abs(P0 - G0)))
    lr0 = opt.param_groups[0]["lr"]

    history["train"].append(float("nan"))
    history["val"].append(va0)
    history["mae_orig"].append(mae0)
    history["lr"].append(lr0)

    best = {
        "mae": mae0,
        "val": va0,
        "per_h": per_h0,
        "ep": 0,
        "state": {k: v.clone().cpu() for k, v in model.state_dict().items()},
    }

    if status_cb is not None:
        status_cb(0, epochs, float("nan"), va0, mae0, lr0)

    for ep in range(1, epochs + 1):
        tr = train_one(
            model,
            train_ld,
            opt,
            device=device,
            use_l1_loss=use_l1_loss,
            smoothl1_beta=smoothl1_beta,
            gate_entropy_lambda=gate_entropy_lambda,
        )

        va, per_h = eval_one(
            model,
            val_ld,
            H,
            D,
            device=device,
            use_l1_loss=use_l1_loss,
            smoothl1_beta=smoothl1_beta,
        )

        sched.step()

        model.eval()
        with torch.no_grad():
            P_list, G_list = [], []
            for xb, yb in val_ld:
                xb = xb.to(device)
                pr, _ = model(xb)
                pr = pr.view(-1, H, D).cpu().numpy()
                yb = yb.view(-1, H, D).cpu().numpy()
                P_list.append(pr)
                G_list.append(yb)

        P = np.vstack(P_list) * sd + mu
        G = np.vstack(G_list) * sd + mu
        mae_orig = float(np.mean(np.abs(P - G)))
        cur_lr = opt.param_groups[0]["lr"]

        history["train"].append(tr)
        history["val"].append(va)
        history["mae_orig"].append(mae_orig)
        history["lr"].append(cur_lr)
        logger.info(
            "ep=%03d | train=%.6f | val=%.6f | mae_orig=%.6f | lr=%.6f",
            ep, tr, va, mae_orig, cur_lr
        )

        if status_cb is not None:
            status_cb(ep, epochs, tr, va, mae_orig, cur_lr)

        if mae_orig < best["mae"] - 1e-6:
            logger.info(
                "New best at ep=%03d | mae_orig=%.6f (old=%.6f)",
                ep, mae_orig, best["mae"]
            )
            best.update(
                {
                    "mae": mae_orig,
                    "val": va,
                    "per_h": per_h,
                    "ep": ep,
                    "state": {k: v.clone().cpu() for k, v in model.state_dict().items()},
                }
            )

        stopper.step(mae_orig)
        if stopper.stop:
            logger.info("Early stopping at ep=%03d", ep)
            break

    model.load_state_dict(best["state"])
    best["epochs_run"] = len(history["val"]) - 1
    best["history"] = history
    logger.info(
                "New best at ep=%03d | mae_orig=%.6f (old=%.6f)",
                ep, mae_orig, best["mae"]
            )
    return best


@torch.no_grad()
def roll_autoregressive(
    model: torch.nn.Module,
    seed_std: np.ndarray,
    H_total: int,
    H: int,
    device: str = "cuda",
) -> np.ndarray:
    model.eval()
    k, D = seed_std.shape
    cur = torch.tensor(seed_std, dtype=torch.float32, device=device)
    outs = []
    produced = 0

    while produced < H_total:
        xb = cur[-k:].unsqueeze(0)
        yb, _ = model(xb)
        yb = yb.view(1, H, D).cpu().numpy()[0]
        take = min(H_total - produced, H)
        outs.append(yb[:take])
        cur = torch.cat(
            [cur, torch.tensor(yb[:take], dtype=torch.float32, device=device)],
            dim=0,
        )
        produced += take

    return np.vstack(outs)

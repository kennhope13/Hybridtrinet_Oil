# forecast_app/train_focus5.py
import numpy as np
import torch
import torch.nn.functional as F


def _horizon_weights_torch(Hh: int, focus_h: int = 5, focus_w: float = 3.0, device=None, dtype=torch.float32):
    Hh = int(Hh)
    focus_h = int(min(max(1, focus_h), Hh))
    w = torch.ones(Hh, device=device, dtype=dtype)
    w[:focus_h] = float(focus_w)
    w = w / (w.mean() + 1e-12)
    return w.view(1, Hh, 1)


def loss_focus5(
    out: torch.Tensor,
    y: torch.Tensor,
    loss_name: str = "huber",
    focus_h: int = 5,
    focus_w: float = 3.0,
    huber_beta: float = 1.0,
) -> torch.Tensor:
    if out.ndim != 3 or y.ndim != 3:
        raise ValueError(f"Expect out,y [B,H,D], got out={tuple(out.shape)} y={tuple(y.shape)}")
    w = _horizon_weights_torch(out.size(1), focus_h, focus_w, device=out.device, dtype=out.dtype)

    ln = str(loss_name).lower()
    if ln == "mae":
        base = (out - y).abs()
    elif ln == "mse":
        base = (out - y).pow(2)
    elif ln == "huber":
        base = F.smooth_l1_loss(out, y, beta=float(huber_beta), reduction="none")
    else:
        raise ValueError("loss_name must be one of: mae | mse | huber")

    return (w * base).mean()


def _batch_xy(batch):
    if isinstance(batch, (tuple, list)) and len(batch) >= 2:
        return batch[0], batch[1]
    if isinstance(batch, dict) and ("x" in batch) and ("y" in batch):
        return batch["x"], batch["y"]
    raise ValueError("Batch format not supported. Expect (x,y) or {'x','y'}")


def _match_out_to_y(out: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    if isinstance(out, (tuple, list)):
        out = out[0]
    if out.ndim == 3:
        if y.ndim == 3 and out.shape[1] == y.shape[2] and out.shape[2] == y.shape[1]:
            return out.transpose(1, 2)
        return out
    if out.ndim == 2 and y.ndim == 3:
        B, HH, DD = y.shape
        if out.shape[1] == HH * DD:
            return out.view(B, HH, DD)
    return out


@torch.no_grad()
def val_mae_real_focus5(model, va_loader, mu, sd, device, focus_h: int = 5, focus_w: float = 3.0):
    model.eval()

    mu = np.asarray(mu, dtype=np.float32).reshape(-1)
    sd = np.asarray(sd, dtype=np.float32).reshape(-1)

    all_sum = 0.0
    all_wsum = 0.0

    for batch in va_loader:
        x, y = _batch_xy(batch)
        x = x.to(device=device, dtype=torch.float32)
        y = y.to(device=device, dtype=torch.float32)

        out = model(x)
        out = _match_out_to_y(out, y)

        if out.ndim != 3:
            raise ValueError(f"Model output must be [B,H,D], got {tuple(out.shape)}")

        B, Hh, Dd = out.shape
        if mu.size < Dd or sd.size < Dd:
            raise ValueError(f"mu/sd shape mismatch: mu={mu.shape}, sd={sd.shape}, D={Dd}")

        mu_t = torch.tensor(mu[:Dd], device=device, dtype=torch.float32).view(1, 1, Dd)
        sd_t = torch.tensor(sd[:Dd], device=device, dtype=torch.float32).view(1, 1, Dd)

        pr = out * sd_t + mu_t
        gt = y * sd_t + mu_t

        err = (pr - gt).abs()
        w_h = _horizon_weights_torch(Hh, focus_h, focus_w, device=device, dtype=err.dtype)

        all_sum += float((w_h * err).sum().item())
        all_wsum += float((w_h * torch.ones_like(err)).sum().item())

    return all_sum / max(1e-12, all_wsum)


def fit_model_better(
    model,
    tr_loader,
    va_loader,
    mu,
    sd,
    epochs: int,
    lr: float,
    loss_name: str = "huber",
    weight_decay: float = 0.0,
    grad_clip: float = 1.0,
    patience: int = 25,
    use_amp: bool = True,
    status_cb=None,
    device: str = "cpu",
    focus_h: int = 5,
    focus_w: float = 3.0,
):
    model = model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=float(lr), weight_decay=float(weight_decay))

    total_steps = max(1, int(epochs) * max(1, len(tr_loader)))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=float(lr), total_steps=total_steps, pct_start=0.15)

    scaler = torch.cuda.amp.GradScaler(enabled=(use_amp and device == "cuda"))

    best_val = float("inf")
    best_state = None
    bad = 0

    for ep in range(1, int(epochs) + 1):
        model.train()
        tr_losses = []

        for batch in tr_loader:
            x, y = _batch_xy(batch)
            x = x.to(device=device, dtype=torch.float32)
            y = y.to(device=device, dtype=torch.float32)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
                out = model(x)
                out = _match_out_to_y(out, y)
                loss = loss_focus5(out, y, loss_name=loss_name, focus_h=focus_h, focus_w=focus_w, huber_beta=1.0)

            scaler.scale(loss).backward()

            if grad_clip and float(grad_clip) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            scaler.step(opt)
            scaler.update()
            sched.step()

            tr_losses.append(float(loss.detach().cpu().item()))

        val_mae = val_mae_real_focus5(model, va_loader, mu=mu, sd=sd, device=device, focus_h=focus_h, focus_w=focus_w)
        tr_loss = float(np.mean(tr_losses)) if tr_losses else float("nan")
        lr_now = float(opt.param_groups[0]["lr"])

        if status_cb is not None:
            status_cb(ep, int(epochs), tr_loss, val_mae, lr_now)

        if val_mae < best_val - 1e-7:
            best_val = val_mae
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
            if bad >= int(patience):
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, best_val

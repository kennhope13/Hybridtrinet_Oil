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


# -------------------------
# Legacy (giữ tương thích)
# -------------------------
def loss_focus5(out, y, loss_name="huber", focus_h=5, focus_w=3.0, huber_beta=1.0):
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


def _match_out_to_y(out, y):
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


def _to_torch_vec(x, device, dtype=torch.float32):
    x = np.asarray(x, dtype=np.float32).reshape(-1)
    return torch.tensor(x, device=device, dtype=dtype)


def _last_price_from_x_std(x_std, tgt_idx, x_mu_tgt, x_sd_tgt):
    tgt_idx = list(map(int, tgt_idx))
    last_std = x_std[:, -1, tgt_idx]               # [B,D]
    last_raw = last_std * x_sd_tgt.view(1, -1) + x_mu_tgt.view(1, -1)
    return last_raw.unsqueeze(1)                   # [B,1,D]


def loss_delta_price_focus(
    out_std, y_std, x_std,
    *,
    tgt_idx, x_mu, x_sd, y_mu, y_sd,
    focus_h=5, focus_w=3.0,
    delta_loss="huber", huber_beta=1.0,
    alpha_delta=0.20, beta_price=1.00,
    eps_mape=1e-3,
):
    if out_std.ndim != 3 or y_std.ndim != 3:
        raise ValueError(f"Expect out,y [B,H,D], got out={tuple(out_std.shape)} y={tuple(y_std.shape)}")

    w_h = _horizon_weights_torch(out_std.size(1), focus_h, focus_w, device=out_std.device, dtype=out_std.dtype)

    ln = str(delta_loss).lower()
    if ln == "mae":
        base = (out_std - y_std).abs()
    elif ln == "mse":
        base = (out_std - y_std).pow(2)
    elif ln == "huber":
        base = F.smooth_l1_loss(out_std, y_std, beta=float(huber_beta), reduction="none")
    else:
        raise ValueError("delta_loss must be one of: huber | mae | mse")

    loss_d = (w_h * base).mean()

    y_mu_t = _to_torch_vec(y_mu, out_std.device, out_std.dtype).view(1, 1, -1)
    y_sd_t = _to_torch_vec(y_sd, out_std.device, out_std.dtype).view(1, 1, -1)
    pred_d = out_std * y_sd_t + y_mu_t
    true_d = y_std   * y_sd_t + y_mu_t

    x_mu_tgt = _to_torch_vec(x_mu, out_std.device, out_std.dtype)[list(map(int, tgt_idx))]
    x_sd_tgt = _to_torch_vec(x_sd, out_std.device, out_std.dtype)[list(map(int, tgt_idx))]
    last_price = _last_price_from_x_std(x_std, tgt_idx, x_mu_tgt, x_sd_tgt)  # [B,1,D]

    pred_p = last_price + torch.cumsum(pred_d, dim=1)
    true_p = last_price + torch.cumsum(true_d, dim=1)

    mape = (pred_p - true_p).abs() / (true_p.abs() + float(eps_mape))
    loss_p = (w_h * mape).mean()

    return float(alpha_delta) * loss_d + float(beta_price) * loss_p


@torch.no_grad()
def val_mae_price_focus(model, va_loader, *, tgt_idx, x_mu, x_sd, y_mu, y_sd, device, focus_h=5, focus_w=3.0):
    model.eval()
    x_mu = np.asarray(x_mu, dtype=np.float32).reshape(-1)
    x_sd = np.asarray(x_sd, dtype=np.float32).reshape(-1)
    y_mu = np.asarray(y_mu, dtype=np.float32).reshape(-1)
    y_sd = np.asarray(y_sd, dtype=np.float32).reshape(-1)

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
        w_h = _horizon_weights_torch(Hh, focus_h, focus_w, device=device, dtype=out.dtype)

        y_mu_t = _to_torch_vec(y_mu[:Dd], device, out.dtype).view(1, 1, Dd)
        y_sd_t = _to_torch_vec(y_sd[:Dd], device, out.dtype).view(1, 1, Dd)
        pred_d = out * y_sd_t + y_mu_t
        true_d = y   * y_sd_t + y_mu_t

        x_mu_tgt = _to_torch_vec(x_mu, device, out.dtype)[list(map(int, tgt_idx))]
        x_sd_tgt = _to_torch_vec(x_sd, device, out.dtype)[list(map(int, tgt_idx))]
        last_price = _last_price_from_x_std(x, tgt_idx, x_mu_tgt, x_sd_tgt)

        pred_p = last_price + torch.cumsum(pred_d, dim=1)
        true_p = last_price + torch.cumsum(true_d, dim=1)

        err = (pred_p - true_p).abs()

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
    # NEW
    use_delta_price_loss: bool = False,
    tgt_idx=None,
    x_mu=None,
    x_sd=None,
    y_mu=None,
    y_sd=None,
    alpha_delta: float = 0.20,
    beta_price: float = 1.00,
    eps_mape: float = 1e-3,
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

                if use_delta_price_loss:
                    loss = loss_delta_price_focus(
                        out, y, x,
                        tgt_idx=tgt_idx, x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd,
                        focus_h=focus_h, focus_w=focus_w,
                        delta_loss=loss_name, huber_beta=1.0,
                        alpha_delta=alpha_delta, beta_price=beta_price,
                        eps_mape=eps_mape,
                    )
                else:
                    loss = loss_focus5(out, y, loss_name=loss_name, focus_h=focus_h, focus_w=focus_w, huber_beta=1.0)

            scaler.scale(loss).backward()

            if grad_clip and float(grad_clip) > 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(grad_clip))

            scaler.step(opt)
            scaler.update()
            sched.step()

            tr_losses.append(float(loss.detach().cpu().item()))

        if use_delta_price_loss:
            val_mae = val_mae_price_focus(
                model, va_loader,
                tgt_idx=tgt_idx, x_mu=x_mu, x_sd=x_sd, y_mu=y_mu, y_sd=y_sd,
                device=device, focus_h=focus_h, focus_w=focus_w
            )
        else:
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
        mu_t = torch.tensor(mu[:Dd], device=device, dtype=torch.float32).view(1, 1, Dd)
        sd_t = torch.tensor(sd[:Dd], device=device, dtype=torch.float32).view(1, 1, Dd)
        pr = out * sd_t + mu_t
        gt = y * sd_t + mu_t

        err = (pr - gt).abs()
        w_h = _horizon_weights_torch(Hh, focus_h, focus_w, device=device, dtype=err.dtype)

        all_sum += float((w_h * err).sum().item())
        all_wsum += float((w_h * torch.ones_like(err)).sum().item())

    return all_sum / max(1e-12, all_wsum)
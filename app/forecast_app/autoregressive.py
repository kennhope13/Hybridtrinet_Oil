# forecast_app/autoregressive.py
import numpy as np
import torch


@torch.no_grad()
def roll_autoregressive_safe(model, seed_std: np.ndarray, H_total: int, H: int, device: str):
    model.eval()

    seed_std = np.asarray(seed_std, dtype=np.float32)
    if seed_std.ndim != 2:
        raise ValueError(f"seed_std must be (K,D), got {seed_std.shape}")

    Kk, D = seed_std.shape
    x = torch.tensor(seed_std, dtype=torch.float32, device=device).unsqueeze(0)  # [1,K,D]

    outs = []
    done = 0

    while done < int(H_total):
        out = model(x)
        if isinstance(out, (tuple, list)):
            out = out[0]

        if out.ndim == 2:
            if out.shape[1] == int(H) * int(D):
                out = out.view(out.shape[0], int(H), int(D))
            else:
                raise ValueError(f"Unexpected 2D out shape: {tuple(out.shape)} (expect [B,H*D])")
        elif out.ndim == 3:
            if out.shape[1] == D and out.shape[2] == int(H):
                out = out.transpose(1, 2)
        else:
            raise ValueError(f"Unexpected out.ndim={out.ndim}, shape={tuple(out.shape)}")

        take = min(int(H), int(H_total) - done)
        chunk = out[:, :take, :]  # [1,take,D]
        outs.append(chunk.detach().cpu())

        x = torch.cat([x, chunk], dim=1)
        x = x[:, -Kk:, :]
        done += take

    y = torch.cat(outs, dim=1).squeeze(0)  # [H_total,D]
    return y.numpy()


def ensure_F_shape(F_std, h_next: int, D: int) -> np.ndarray:
    if isinstance(F_std, torch.Tensor):
        F_std = F_std.detach().cpu().numpy()
    F_std = np.asarray(F_std)

    if F_std.ndim == 1:
        F_std = F_std.reshape(-1, 1)
    elif F_std.ndim == 3:
        F_std = F_std.reshape(-1, F_std.shape[-1])

    if F_std.shape[0] == D and F_std.shape[1] == h_next:
        F_std = F_std.T

    if F_std.shape[0] != h_next or F_std.shape[1] != D:
        raise ValueError(f"Forecast shape mismatch. Expect ({h_next},{D}), got {F_std.shape}")
    return F_std

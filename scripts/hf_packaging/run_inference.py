import argparse
import json
import os
import sys
from pathlib import Path
import numpy as np
import torch

try:
    from model_architecture import HybridTriNet
except ImportError:
    from scripts.hf_packaging.model_architecture import HybridTriNet


def load_forecasting_pipeline(horizon_dir: str, device: str = "cpu"):
    """
    Loads model, config, feature metadata, and scalers for a specific horizon.
    """
    h_path = Path(horizon_dir)
    cfg_path = h_path / "model_config.json"
    feat_path = h_path / "feature_cols.json"
    model_path = h_path / "model.pt"

    if not cfg_path.exists() or not feat_path.exists() or not model_path.exists():
        raise FileNotFoundError(f"Missing required model files in {horizon_dir}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(feat_path, "r", encoding="utf-8") as f:
        feat_meta = json.load(f)

    x_mu = np.load(h_path / "x_mu.npy").astype(np.float32)
    x_sd = np.load(h_path / "x_sd.npy").astype(np.float32)
    y_mu = np.load(h_path / "y_mu.npy").astype(np.float32)
    y_sd = np.load(h_path / "y_sd.npy").astype(np.float32)

    model = HybridTriNet(
        k=config["K"],
        H=config["H"],
        D_in=config["D_in"],
        D_out=config["D_out"],
        d_feat=config.get("d_feat", 96),
        kan_M=config.get("kan_M", 8),
        kan_depth=config.get("kan_depth", 2),
        kan_drop=config.get("kan_drop", 0.2),
        gru_hidden=config.get("gru_hidden", 128),
        gru_layers=config.get("gru_layers", 1),
        gru_drop=config.get("gru_drop", 0.1),
        attn_dmodel=config.get("attn_dmodel", 64),
        attn_heads=config.get("attn_heads", 4),
        attn_layers=config.get("attn_layers", 2),
        attn_drop=config.get("attn_drop", 0.2),
        patch_len=config.get("patch_len", 16),
        stride=config.get("stride", 8),
    ).to(device)

    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.eval()

    bundle = {
        "model": model,
        "config": config,
        "feature_meta": feat_meta,
        "feature_cols": feat_meta["feature_cols"],
        "target_cols": [feat_meta["feature_cols"][i] for i in feat_meta["tgt_idx"]],
        "x_mu": x_mu,
        "x_sd": x_sd,
        "y_mu": y_mu,
        "y_sd": y_sd,
        "device": device,
    }
    return bundle


def forecast(bundle: dict, raw_input_window: np.ndarray) -> np.ndarray:
    """
    Runs forward inference and inverse standardization.

    raw_input_window: numpy array of shape (B, K, D_in) or (K, D_in) in raw feature scale.
    returns: numpy array of shape (B, H, D_out) in original target units defined by the dataset metadata.
    """
    if raw_input_window.ndim == 2:
        raw_input_window = np.expand_dims(raw_input_window, axis=0)

    B, K, D_in = raw_input_window.shape
    expected_K = bundle["config"]["K"]
    expected_D = bundle["config"]["D_in"]
    if K != expected_K or D_in != expected_D:
        raise ValueError(f"Input shape mismatch. Expected (B, {expected_K}, {expected_D}), got {raw_input_window.shape}")

    # Standardize input
    x_mu = bundle["x_mu"]
    x_sd = bundle["x_sd"]
    x_norm = (raw_input_window.astype(np.float32) - x_mu) / (x_sd + 1e-8)

    model = bundle["model"]
    device = bundle["device"]

    with torch.no_grad():
        x_tensor = torch.from_numpy(x_norm).to(device)
        out_tensor, _ = model(x_tensor)
        out_norm = out_tensor.cpu().numpy().reshape(B, bundle["config"]["H"], bundle["config"]["D_out"])

    # Inverse standardization to original target units
    y_mu = bundle["y_mu"]
    y_sd = bundle["y_sd"]
    out_real = out_norm * (y_sd + 1e-8) + y_mu
    return out_real


def main():
    parser = argparse.ArgumentParser(description="Run standalone inference with HybridTriNet bundle.")
    parser.add_argument("--model_dir", type=str, default="models/h60", help="Path to horizon directory (e.g. models/h60)")
    parser.add_argument("--device", type=str, default="cpu", help="Device to run inference on (cpu/cuda)")
    args = parser.parse_args()

    print(f"[INFO] Loading pipeline from: {args.model_dir} (device={args.device})")
    bundle = load_forecasting_pipeline(args.model_dir, device=args.device)

    K = bundle["config"]["K"]
    D_in = bundle["config"]["D_in"]
    H = bundle["config"]["H"]
    D_out = bundle["config"]["D_out"]
    targets = bundle["target_cols"]

    print(f"[INFO] Pipeline loaded: Horizon H={H}, Lookback K={K}, D_in={D_in}, Targets={targets}")

    # Generate synthetic input window matching mean/variance scale
    synthetic_input = np.random.normal(loc=bundle["x_mu"], scale=bundle["x_sd"], size=(1, K, D_in)).astype(np.float32)
    predictions = forecast(bundle, synthetic_input)

    print(f"[SUCCESS] Inference complete. Output shape: {predictions.shape}")
    print(f"[INFO] Sample forecast (step 0 to min(5, H)) in original target units defined by dataset metadata:")
    for t_idx, t_name in enumerate(targets):
        steps = predictions[0, :min(5, H), t_idx]
        print(f"  - {t_name:12s}: {np.round(steps, 2).tolist()}")


if __name__ == "__main__":
    main()

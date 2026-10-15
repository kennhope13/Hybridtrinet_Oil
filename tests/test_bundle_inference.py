from pathlib import Path
import numpy as np
import pytest
import torch

from inference.model_architecture import HybridTriNet
from inference.run_inference import load_forecasting_pipeline, forecast
from inference.verify_bundle import verify_bundle

BUNDLE_DIR = Path(__file__).resolve().parent.parent / "hf_releases" / "hybridtrinet-oil-forecasting"


def test_model_architecture_shapes():
    B = 2
    K = 64
    D_in = 15
    D_out = 4
    H = 10

    model = HybridTriNet(
        k=K, H=H, D_in=D_in, D_out=D_out,
        d_feat=96, kan_M=8, kan_depth=2,
        gru_hidden=128, gru_layers=1,
        attn_dmodel=64, attn_heads=4, attn_layers=2,
        patch_len=16, stride=8,
    )
    model.eval()

    dummy_input = torch.randn(B, K, D_in)
    with torch.no_grad():
        out, (yk, yg, ya, w) = model(dummy_input)

    assert out.shape == (B, H * D_out)
    assert yk.shape == (B, H, D_out)
    assert yg.shape == (B, H, D_out)
    assert ya.shape == (B, H, D_out)
    assert w.shape == (B, H, 3)
    # Check softmax partition of unity
    assert torch.allclose(w.sum(dim=-1), torch.ones(B, H), atol=1e-5)


def test_bundle_integrity():
    if not BUNDLE_DIR.exists():
        pytest.skip(f"Bundle directory {BUNDLE_DIR} does not exist.")
    # Run full verify_bundle logic
    verify_bundle(str(BUNDLE_DIR))


def test_reproducible_forecast_execution():
    h60_dir = BUNDLE_DIR / "models" / "h60"
    if not h60_dir.exists():
        pytest.skip(f"Model directory {h60_dir} not found.")

    bundle = load_forecasting_pipeline(str(h60_dir), device="cpu")
    K = bundle["config"]["K"]
    D_in = bundle["config"]["D_in"]
    H = bundle["config"]["H"]
    D_out = bundle["config"]["D_out"]

    # Fixed synthetic input
    np.random.seed(42)
    sample_input = np.random.normal(loc=bundle["x_mu"], scale=bundle["x_sd"], size=(1, K, D_in)).astype(np.float32)

    pred1 = forecast(bundle, sample_input)
    pred2 = forecast(bundle, sample_input)

    assert pred1.shape == (1, H, D_out)
    assert not np.isnan(pred1).any()
    assert not np.isinf(pred1).any()
    np.testing.assert_allclose(pred1, pred2, atol=1e-6)

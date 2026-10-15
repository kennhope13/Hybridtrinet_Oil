# -*- coding: utf-8 -*-
"""
High-fidelity Isometric / 3D Volumetric Tensor Block Visualization for GUMNet and Dual-Model Comparison.
Renders:
- docs/figures/8_gumnet_overview.png
- docs/figures/9_gumnet_wavelet_kan.png
- docs/figures/10_gumnet_dilated_cnn.png
- docs/figures/11_gumnet_gru_attention.png
- docs/figures/12_gumnet_adaptive_gating.png
- docs/figures/13_gumnet_quantile_output.png
- docs/figures/14_model_comparison.png
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Actual GUMNet hyperparameters verified from code and checkpoint
SEQ_LEN = 30
INPUT_DIM = 17
OUTPUT_DIM = 4
D_FEAT = 64
NUM_QUANTILES = 3
QUANTILES = [0.1, 0.5, 0.9]


def draw_isometric_cube(ax, x, y, w, h, d, face_color, edge_color="#334155", alpha=1.0, zorder=2):
    dx = d * 0.7
    dy = d * 0.5
    c_rgb = np.array(plt.matplotlib.colors.to_rgb(face_color))
    c_top = tuple(np.clip(c_rgb * 1.25, 0, 1))
    c_side = tuple(np.clip(c_rgb * 0.75, 0, 1))
    c_front = tuple(c_rgb)

    front = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True, facecolor=c_front, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=zorder)
    ax.add_patch(front)

    top = Polygon([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)], closed=True, facecolor=c_top, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=zorder + 1)
    ax.add_patch(top)

    side = Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True, facecolor=c_side, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=zorder + 1)
    ax.add_patch(side)


def save_fig(fig, filepath):
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Rendered: {filepath}")


# =============================================================================
# 8. GUMNet Architecture Overview
# =============================================================================
def draw_8_gumnet_overview():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8.0, 8.6, "8. GUMNet Architecture (3D Volumetric Overview)", fontsize=16, fontweight="bold", color="#0f172a", ha="center")
    ax.text(8.0, 8.2, "Quantile Probabilistic Forecasting | Wavelet-KAN + Dilated CNN + GRU-Attention with Adaptive Softmax Gating", fontsize=10.5, color="#64748b", ha="center")

    # Input Tensor [B, 30, 17]
    draw_isometric_cube(ax, 0.8, 3.5, 1.8, 2.2, 0.8, "#bae6fd", edge_color="#0284c7")
    ax.text(1.7, 6.2, "Input Tensor X\n[B, 30, 17]", fontsize=10, fontweight="bold", color="#0369a1", ha="center")
    ax.text(1.7, 3.0, "seq_len = 30 steps\ninput_dim = 17 features", fontsize=8, color="#334155", ha="center")

    # Branch 1: Dilated CNN (Top)
    draw_isometric_cube(ax, 4.5, 5.8, 2.6, 1.4, 0.6, "#fed7aa", edge_color="#ea580c")
    ax.text(5.8, 7.5, "Branch 1: Dilated CNN\n[B, 17, 30] -> [B, 64]", fontsize=9.5, fontweight="bold", color="#9a3412", ha="center")
    ax.text(5.8, 5.3, "Conv1d(k=3, pad=2, dil=2) + AvgPool1d(1)\nFeature Vector f_cnn: [B, 64]", fontsize=7.8, color="#7c2d12", ha="center")

    # Branch 2: GRU + Attention (Middle)
    draw_isometric_cube(ax, 4.5, 3.6, 2.6, 1.4, 0.6, "#86efac", edge_color="#16a34a")
    ax.text(5.8, 5.2, "Branch 2: GRU + MultiHeadAttn\n[B, 30, 17] -> [B, 64]", fontsize=9.5, fontweight="bold", color="#14532d", ha="center")
    ax.text(5.8, 3.1, "2-layer GRU (drop=0.1) + MultiHeadAttn(4 heads)\nFeature Vector f_gru: [B, 64]", fontsize=7.8, color="#166534", ha="center")

    # Branch 3: Wavelet-KAN (Bottom)
    draw_isometric_cube(ax, 4.5, 1.4, 2.6, 1.4, 0.6, "#fde047", edge_color="#ca8a04")
    ax.text(5.8, 2.9, "Branch 3: Wavelet-KAN\n[B, 30, 17] -> [B, 64]", fontsize=9.5, fontweight="bold", color="#854d0e", ha="center")
    ax.text(5.8, 0.9, "Mexican Hat Wavelet basis + AvgPool1d(1)\nFeature Vector f_kan: [B, 64]", fontsize=7.8, color="#713f12", ha="center")

    # Feature Concat & Gating
    draw_isometric_cube(ax, 8.8, 2.8, 2.4, 3.0, 0.8, "#d8b4fe", edge_color="#7e22ce")
    ax.text(10.0, 6.3, "Adaptive Softmax Gating\nConcat: [B, 192]", fontsize=10, fontweight="bold", color="#581c87", ha="center")
    ax.text(10.0, 3.9, "f_concat = [f_cnn, f_gru, f_kan]\nLinear(192, 32) + ReLU + Linear(32, 3)\nSoftmax weights: w in [B, 3]\nw_cnn + w_gru + w_kan = 1.0\nf_fused = sum(w_i * f_i) in [B, 64]", fontsize=7.6, color="#3b0764", ha="center")

    # Quantile Output Head
    draw_isometric_cube(ax, 12.8, 3.2, 2.0, 2.5, 0.8, "#fbcfe8", edge_color="#db2777")
    ax.text(13.8, 6.2, "Quantile Output Head\n[B, H, 4, 3]", fontsize=10.5, fontweight="bold", color="#9d174d", ha="center")
    ax.text(13.8, 3.4, "nn.Linear(64, H * 4 * 3)\nQuantiles: q=0.10, 0.50, 0.90\nUncertainty Bound\n4 Commodity Targets", fontsize=7.8, color="#831843", ha="center")

    # Arrows
    arrow_props = dict(arrowstyle="-|>", color="#475569", lw=2.0, mutation_scale=15)
    ax.annotate("", xy=(4.4, 6.5), xytext=(2.9, 4.8), arrowprops=arrow_props)
    ax.annotate("", xy=(4.4, 4.3), xytext=(2.9, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(4.4, 2.1), xytext=(2.9, 4.2), arrowprops=arrow_props)

    arrow_purple = dict(arrowstyle="-|>", color="#7e22ce", lw=2.0, mutation_scale=15)
    ax.annotate("", xy=(8.7, 5.0), xytext=(7.4, 6.5), arrowprops=arrow_purple)
    ax.annotate("", xy=(8.7, 4.3), xytext=(7.4, 4.3), arrowprops=arrow_purple)
    ax.annotate("", xy=(8.7, 3.6), xytext=(7.4, 2.1), arrowprops=arrow_purple)

    ax.annotate("", xy=(12.7, 4.5), xytext=(11.5, 4.5), arrowprops=dict(arrowstyle="-|>", color="#db2777", lw=2.5, mutation_scale=18))

    save_fig(fig, FIG_DIR / "8_gumnet_overview.png")


# =============================================================================
# 9. GUMNet Wavelet-KAN Branch
# =============================================================================
def draw_9_gumnet_wavelet_kan():
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(7.5, 6.0, "9. GUMNet Branch 1: Wavelet-KAN (Mexican Hat Wavelet Transformation)", fontsize=15, fontweight="bold", color="#854d0e", ha="center")
    ax.text(7.5, 5.65, "Learnable translation and scale parameters parameterizing continuous Mexican Hat wavelets for multi-scale frequency localization", fontsize=10, color="#713f12", ha="center")

    blocks = [
        {"x": 0.8, "y": 2.0, "w": 1.8, "h": 2.6, "d": 0.7, "color": "#bae6fd", "edge": "#0284c7", "title": "Sequence Input", "dim": "[B, 30, 17]", "desc": "seq_len = 30\ninput_dim = 17"},
        {"x": 3.6, "y": 1.8, "w": 2.6, "h": 3.0, "d": 0.8, "color": "#fde047", "edge": "#ca8a04", "title": "Wavelet Normalization", "dim": "x_norm in [B, 30, 17]", "desc": "scale, translation in (1, 1, 17)\nx_norm = (x - trans) / (scale + 1e-8)\nLearnable parameter shift"},
        {"x": 7.2, "y": 1.6, "w": 3.2, "h": 3.4, "d": 0.8, "color": "#facc15", "edge": "#854d0e", "title": "Mexican Hat + Base", "dim": "kan_out in [B, 30, 64]", "desc": "psi = (1 - x_norm^2) * exp(-0.5 * x_norm^2)\nbase = SiLU(Linear1(x))\nwav = Linear2(psi)\nout = base + wav"},
        {"x": 11.4, "y": 2.2, "w": 2.2, "h": 2.2, "d": 0.6, "color": "#fef08a", "edge": "#ca8a04", "title": "Adaptive AvgPool", "dim": "f_kan: [B, 64]", "desc": "Transpose (1, 2) -> (B, 64, 30)\nAdaptiveAvgPool1d(1)\nSqueeze -> [B, 64]"},
    ]

    for b in blocks:
        draw_isometric_cube(ax, b["x"], b["y"], b["w"], b["h"], b["d"], b["color"], edge_color=b["edge"])
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.65, b["title"], fontsize=9.5, fontweight="bold", color="#854d0e", ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.35, b["dim"], fontsize=8.5, fontweight="bold", color=b["edge"], ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] - 0.7, b["desc"], fontsize=7.5, color="#713f12", ha="center", va="top")

    arrow_props = dict(arrowstyle="-|>", color="#ca8a04", lw=2.0, mutation_scale=15)
    for i in range(len(blocks) - 1):
        x1 = blocks[i]["x"] + blocks[i]["w"] + blocks[i]["d"] * 0.7 + 0.05
        x2 = blocks[i + 1]["x"] - 0.05
        y_mid = blocks[i]["y"] + blocks[i]["h"] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "9_gumnet_wavelet_kan.png")


# =============================================================================
# 10. GUMNet Dilated CNN Branch
# =============================================================================
def draw_10_gumnet_dilated_cnn():
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(7.5, 6.0, "10. GUMNet Branch 2: Dilated 1D Convolution (Expanded Receptive Field)", fontsize=15, fontweight="bold", color="#9a3412", ha="center")
    ax.text(7.5, 5.65, "Dilated causal convolution captures local multiscale volatility patterns without increasing parameter count", fontsize=10, color="#7c2d12", ha="center")

    blocks = [
        {"x": 0.8, "y": 2.0, "w": 1.8, "h": 2.6, "d": 0.7, "color": "#bae6fd", "edge": "#0284c7", "title": "Sequence Input", "dim": "[B, 30, 17]", "desc": "seq_len = 30\ninput_dim = 17"},
        {"x": 3.6, "y": 2.1, "w": 2.0, "h": 2.4, "d": 0.6, "color": "#fed7aa", "edge": "#ea580c", "title": "Transpose Channels", "dim": "x_cnn: [B, 17, 30]", "desc": "Transpose dim 1 & 2\nChannels = 17\nLength = 30"},
        {"x": 6.8, "y": 1.7, "w": 3.4, "h": 3.2, "d": 0.8, "color": "#fdba74", "edge": "#c2410c", "title": "Dilated Conv1d + ReLU", "dim": "conv_out: [B, 64, 30]", "desc": "in_channels=17, out_channels=64\nkernel_size=3, padding=2\ndilation=2 (receptive field = 5)\nReLU non-linear activation"},
        {"x": 11.2, "y": 2.2, "w": 2.4, "h": 2.2, "d": 0.6, "color": "#fed7aa", "edge": "#ea580c", "title": "Adaptive AvgPool", "dim": "f_cnn: [B, 64]", "desc": "AdaptiveAvgPool1d(1)\nSqueeze last dim\nFeature representation"},
    ]

    for b in blocks:
        draw_isometric_cube(ax, b["x"], b["y"], b["w"], b["h"], b["d"], b["color"], edge_color=b["edge"])
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.65, b["title"], fontsize=9.5, fontweight="bold", color="#9a3412", ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.35, b["dim"], fontsize=8.5, fontweight="bold", color=b["edge"], ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] - 0.7, b["desc"], fontsize=7.5, color="#7c2d12", ha="center", va="top")

    arrow_props = dict(arrowstyle="-|>", color="#ea580c", lw=2.0, mutation_scale=15)
    for i in range(len(blocks) - 1):
        x1 = blocks[i]["x"] + blocks[i]["w"] + blocks[i]["d"] * 0.7 + 0.05
        x2 = blocks[i + 1]["x"] - 0.05
        y_mid = blocks[i]["y"] + blocks[i]["h"] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "10_gumnet_dilated_cnn.png")


# =============================================================================
# 11. GUMNet GRU + Multi-Head Attention Branch
# =============================================================================
def draw_11_gumnet_gru_attention():
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(7.5, 6.0, "11. GUMNet Branch 3: 2-Layer GRU + Multi-Head Attention", fontsize=15, fontweight="bold", color="#14532d", ha="center")
    ax.text(7.5, 5.65, "Stacked recurrent gating followed by multi-head self-attention extracting temporal cross-step dependencies", fontsize=10, color="#166534", ha="center")

    blocks = [
        {"x": 0.8, "y": 2.0, "w": 1.8, "h": 2.6, "d": 0.7, "color": "#bae6fd", "edge": "#0284c7", "title": "Sequence Input", "dim": "[B, 30, 17]", "desc": "seq_len = 30\ninput_dim = 17"},
        {"x": 3.7, "y": 1.7, "w": 2.8, "h": 3.2, "d": 0.8, "color": "#86efac", "edge": "#16a34a", "title": "2-Layer Recurrent GRU", "dim": "gru_out: [B, 30, 64]", "desc": "in=17, hidden=64\nnum_layers=2\ndropout=0.1, batch_first=True"},
        {"x": 7.6, "y": 1.7, "w": 3.2, "h": 3.2, "d": 0.8, "color": "#4ade80", "edge": "#15803d", "title": "Multi-Head Attention", "dim": "attn_out: [B, 30, 64]", "desc": "embed_dim=64, num_heads=4\nSelf-Attention (Q=K=V=gru_out)\nbatch_first=True"},
        {"x": 11.8, "y": 2.2, "w": 2.2, "h": 2.2, "d": 0.6, "color": "#bbf7d0", "edge": "#14532d", "title": "Terminal Context", "dim": "f_gru: [B, 64]", "desc": "Extract last step:\nattn_out[:, -1, :]\nFinal temporal feature"},
    ]

    for b in blocks:
        draw_isometric_cube(ax, b["x"], b["y"], b["w"], b["h"], b["d"], b["color"], edge_color=b["edge"])
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.65, b["title"], fontsize=9.5, fontweight="bold", color="#14532d", ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.35, b["dim"], fontsize=8.5, fontweight="bold", color=b["edge"], ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] - 0.7, b["desc"], fontsize=7.5, color="#052e16", ha="center", va="top")

    arrow_props = dict(arrowstyle="-|>", color="#15803d", lw=2.0, mutation_scale=15)
    for i in range(len(blocks) - 1):
        x1 = blocks[i]["x"] + blocks[i]["w"] + blocks[i]["d"] * 0.7 + 0.05
        x2 = blocks[i + 1]["x"] - 0.05
        y_mid = blocks[i]["y"] + blocks[i]["h"] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "11_gumnet_gru_attention.png")


# =============================================================================
# 12. GUMNet Adaptive Gating
# =============================================================================
def draw_12_gumnet_adaptive_gating():
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(7.5, 7.0, "12. GUMNet Adaptive Softmax Gating Mechanism", fontsize=16, fontweight="bold", color="#581c87", ha="center")
    ax.text(7.5, 6.6, "Learned dynamic fusion balancing local CNN edges, sequential GRU momentum, and Wavelet-KAN non-linearity", fontsize=10, color="#6b21a8", ha="center")

    draw_isometric_cube(ax, 0.8, 4.8, 1.8, 1.3, 0.5, "#fed7aa", edge_color="#ea580c")
    ax.text(1.7, 6.4, "f_cnn: [B, 64]\nDilated Conv1D", fontsize=8.5, fontweight="bold", color="#9a3412", ha="center")

    draw_isometric_cube(ax, 0.8, 3.1, 1.8, 1.3, 0.5, "#86efac", edge_color="#16a34a")
    ax.text(1.7, 4.7, "f_gru: [B, 64]\nGRU + Attention", fontsize=8.5, fontweight="bold", color="#14532d", ha="center")

    draw_isometric_cube(ax, 0.8, 1.4, 1.8, 1.3, 0.5, "#fde047", edge_color="#ca8a04")
    ax.text(1.7, 3.0, "f_kan: [B, 64]\nWavelet-KAN", fontsize=8.5, fontweight="bold", color="#854d0e", ha="center")

    # Feature Concat [B, 192]
    draw_isometric_cube(ax, 3.9, 2.5, 2.0, 2.6, 0.7, "#e9d5ff", edge_color="#a855f7")
    ax.text(4.9, 5.5, "Concat Features\n[B, 192]", fontsize=10, fontweight="bold", color="#581c87", ha="center")
    ax.text(4.9, 2.0, "f_concat =\n[f_cnn, f_gru, f_kan]", fontsize=8, color="#6b21a8", ha="center")

    # Gating Network
    draw_isometric_cube(ax, 7.1, 2.3, 2.4, 3.0, 0.8, "#c084fc", edge_color="#7e22ce")
    ax.text(8.3, 5.7, "Gating MLP\n192 -> 32 -> 3", fontsize=10, fontweight="bold", color="#3b0764", ha="center")
    ax.text(8.3, 3.8, "Linear(192, 32)\nReLU()\nLinear(32, 3)\nSoftmax(dim=-1)", fontsize=8, color="#2e1065", ha="center")
    ax.text(8.3, 1.8, "Weights: w in [B, 3]", fontsize=9, fontweight="bold", color="#581c87", ha="center")

    # Fused Feature
    draw_isometric_cube(ax, 10.6, 2.5, 3.6, 2.6, 0.8, "#d8b4fe", edge_color="#7e22ce")
    ax.text(12.4, 5.5, "Fused Feature Vector\nf_fused in [B, 64]", fontsize=10.5, fontweight="bold", color="#581c87", ha="center")
    ax.text(12.4, 4.1, "f_fused = (w_cnn * f_cnn) +\n          (w_gru * f_gru) +\n          (w_kan * f_kan)\nw_cnn + w_gru + w_kan = 1.0 (Exact)", fontsize=8.5, color="#3b0764", ha="center")
    ax.text(12.4, 2.9, "Feeds directly to Quantile Output Head", fontsize=8.5, fontweight="bold", color="#9333ea", ha="center")

    arrow_purple = dict(arrowstyle="-|>", color="#7e22ce", lw=2.0, mutation_scale=15)
    ax.annotate("", xy=(3.8, 4.3), xytext=(2.8, 5.3), arrowprops=arrow_purple)
    ax.annotate("", xy=(3.8, 3.8), xytext=(2.8, 3.7), arrowprops=arrow_purple)
    ax.annotate("", xy=(3.8, 3.3), xytext=(2.8, 2.0), arrowprops=arrow_purple)

    ax.annotate("", xy=(7.0, 3.8), xytext=(6.1, 3.8), arrowprops=arrow_purple)
    ax.annotate("", xy=(10.5, 3.8), xytext=(9.7, 3.8), arrowprops=arrow_purple)

    save_fig(fig, FIG_DIR / "12_gumnet_adaptive_gating.png")


# =============================================================================
# 13. GUMNet Quantile Output Head
# =============================================================================
def draw_13_gumnet_quantile_output():
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(7.5, 7.0, "13. GUMNet Quantile Output Head & Uncertainty Intervals", fontsize=16, fontweight="bold", color="#9d174d", ha="center")
    ax.text(7.5, 6.6, "Direct multi-quantile regression (q=0.10, 0.50, 0.90) generating probabilistic 80% coverage intervals", fontsize=10, color="#831843", ha="center")

    # Fused Feature In
    draw_isometric_cube(ax, 0.8, 2.8, 2.0, 2.4, 0.6, "#d8b4fe", edge_color="#7e22ce")
    ax.text(1.8, 5.6, "Fused Vector\nf_fused in [B, 64]", fontsize=9.5, fontweight="bold", color="#581c87", ha="center")
    ax.text(1.8, 2.4, "Output of Gating", fontsize=8, color="#3b0764", ha="center")

    # Linear Projection
    draw_isometric_cube(ax, 4.0, 2.6, 2.4, 2.8, 0.7, "#fbcfe8", edge_color="#db2777")
    ax.text(5.2, 5.8, "Linear Projection\n64 -> H * 4 * 3", fontsize=10, fontweight="bold", color="#9d174d", ha="center")
    ax.text(5.2, 3.8, "nn.Linear(64, H*4*3)\nReshape to:\n[B, H, 4, 3]", fontsize=8.5, color="#831843", ha="center")

    # 3 Quantiles Visual Cards
    q_cards = [
        {"y": 4.8, "q": "q = 0.90", "name": "Upper Bound (P90)", "desc": "90th percentile pessimistic forecast / peak price risk", "bg": "#fce7f3", "edge": "#be185d"},
        {"y": 3.0, "q": "q = 0.50", "name": "Median Forecast (P50)", "desc": "Point-forecast median / primary benchmark evaluation", "bg": "#fbcfe8", "edge": "#db2777"},
        {"y": 1.2, "q": "q = 0.10", "name": "Lower Bound (P10)", "desc": "10th percentile optimistic forecast / downside price risk", "bg": "#fdf2f8", "edge": "#9d174d"},
    ]

    for c in q_cards:
        draw_isometric_cube(ax, 7.8, c["y"], 6.4, 1.4, 0.5, c["bg"], edge_color=c["edge"])
        ax.text(8.8, c["y"] + 0.7, c["q"], fontsize=12, fontweight="bold", color=c["edge"], va="center")
        ax.text(10.2, c["y"] + 0.85, c["name"], fontsize=9.5, fontweight="bold", color="#0f172a", va="center")
        ax.text(10.2, c["y"] + 0.45, c["desc"], fontsize=8, color="#334155", va="center")

    # Arrows
    ax.annotate("", xy=(3.9, 4.0), xytext=(3.0, 4.0), arrowprops=dict(arrowstyle="-|>", color="#db2777", lw=2.2, mutation_scale=15))
    ax.annotate("", xy=(7.7, 5.5), xytext=(6.6, 4.5), arrowprops=dict(arrowstyle="-|>", color="#be185d", lw=2.0, mutation_scale=14))
    ax.annotate("", xy=(7.7, 3.7), xytext=(6.6, 4.0), arrowprops=dict(arrowstyle="-|>", color="#db2777", lw=2.0, mutation_scale=14))
    ax.annotate("", xy=(7.7, 1.9), xytext=(6.6, 3.5), arrowprops=dict(arrowstyle="-|>", color="#9d174d", lw=2.0, mutation_scale=14))

    save_fig(fig, FIG_DIR / "13_gumnet_quantile_output.png")


# =============================================================================
# 14. Model Comparison Diagram (GUMNet vs HybridTriNet)
# =============================================================================
def draw_14_model_comparison():
    fig, ax = plt.subplots(figsize=(16, 9.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.5)
    ax.axis("off")

    ax.text(8.0, 9.0, "14. System Model Comparison: GUMNet vs. HybridTriNet", fontsize=16, fontweight="bold", color="#0f172a", ha="center")
    ax.text(8.0, 8.6, "Comparative Architectural Paradigm, Objective Function, Target Handling & Operational Roles", fontsize=10.5, color="#64748b", ha="center")

    # Column 1: GUMNet Panel
    ax.add_patch(patches.FancyBboxPatch((0.6, 0.5), 7.1, 7.8, boxstyle="round,pad=0.2", facecolor="#fdf2f8", edgecolor="#db2777", lw=2.2))
    ax.text(4.15, 8.0, "GUMNet (Production & Backtest Pipeline)", fontsize=13, fontweight="bold", color="#9d174d", ha="center")
    ax.text(4.15, 7.6, "Probabilistic Multi-Horizon Quantile Forecaster", fontsize=9.5, style="italic", color="#be185d", ha="center")

    draw_isometric_cube(ax, 1.2, 5.6, 1.5, 1.4, 0.4, "#fbcfe8", edge_color="#db2777")
    ax.text(1.95, 6.3, "Input\n[B, 30, 17]", fontsize=8.5, fontweight="bold", color="#9d174d", ha="center")

    draw_isometric_cube(ax, 3.4, 5.6, 2.0, 1.4, 0.4, "#f472b6", edge_color="#be185d")
    ax.text(4.4, 6.3, "Tri-Branch\nWavelet+CNN+GRU", fontsize=8.5, fontweight="bold", color="#831843", ha="center")

    draw_isometric_cube(ax, 5.9, 5.6, 1.5, 1.4, 0.4, "#db2777", edge_color="#9d174d")
    ax.text(6.65, 6.3, "Output\n[B, H, 4, 3]", fontsize=8.5, fontweight="bold", color="#ffffff", ha="center")

    gum_specs = [
        ("Operational Role", "Primary operational pipeline model / Candidate promotion backtester"),
        ("Forecasting Paradigm", "Probabilistic Quantile Forecast (q = 0.10, 0.50, 0.90)"),
        ("Objective Loss", "Pinball Quantile Loss (Multi-quantile asymmetric penalty)"),
        ("Lookback Window", "seq_len = 30 trading steps"),
        ("Input Features", "17 market & exogenous features"),
        ("Representation Branches", "1. Wavelet-KAN (Mexican Hat continuous basis)\n2. Dilated 1D CNN (dilation=2, k=3)\n3. 2-Layer GRU (hidden=64) + Multi-Head Attention"),
        ("Gating Mechanism", "Softmax vector w: [B, 3] fusing feature representations"),
        ("Output Dimension", "[Batch, Horizon, 4 targets, 3 quantiles]"),
    ]
    y_pos = 4.8
    for label, val in gum_specs:
        ax.text(1.0, y_pos, f"• {label}:", fontsize=8.8, fontweight="bold", color="#831843")
        ax.text(1.2, y_pos - 0.28, val, fontsize=8.0, color="#1e293b")
        y_pos -= 0.55

    # Column 2: HybridTriNet Panel
    ax.add_patch(patches.FancyBboxPatch((8.3, 0.5), 7.1, 7.8, boxstyle="round,pad=0.2", facecolor="#eff6ff", edgecolor="#2563eb", lw=2.2))
    ax.text(11.85, 8.0, "HybridTriNet (Clean Hub Packaging Release)", fontsize=13, fontweight="bold", color="#1e40af", ha="center")
    ax.text(11.85, 7.6, "Deterministic Deep Multi-Horizon Point Forecaster", fontsize=9.5, style="italic", color="#2563eb", ha="center")

    draw_isometric_cube(ax, 8.9, 5.6, 1.5, 1.4, 0.4, "#bfdbfe", edge_color="#2563eb")
    ax.text(9.65, 6.3, "Input\n[B, 64, 15]", fontsize=8.5, fontweight="bold", color="#1e40af", ha="center")

    draw_isometric_cube(ax, 11.1, 5.6, 2.0, 1.4, 0.4, "#60a5fa", edge_color="#1d4ed8")
    ax.text(12.1, 6.3, "Tri-Branch\nKAN+GRU+PatchAttn", fontsize=8.5, fontweight="bold", color="#172554", ha="center")

    draw_isometric_cube(ax, 13.6, 5.6, 1.5, 1.4, 0.4, "#2563eb", edge_color="#1e3a8a")
    ax.text(14.35, 6.3, "Output\n[B, H, 4]", fontsize=8.5, fontweight="bold", color="#ffffff", ha="center")

    hyb_specs = [
        ("Operational Role", "Stand-alone clean 15-feature release packaged for Hugging Face Hub"),
        ("Forecasting Paradigm", "Deterministic Point Forecast with horizon-specific scalers"),
        ("Objective Loss", "Smooth L1 Loss (Huber Loss variant for outlier robustness)"),
        ("Lookback Window", "K = 64 trading steps (Extended history)"),
        ("Input Features", "15 clean market features (synthetic temporal index purged)"),
        ("Representation Branches", "1. KAN (Learnable RBF M=8, 2x KANBlocks)\n2. Sequential GRU (hidden=128, terminal state)\n3. Patch Transformer (7 patches, 4 heads, 2 layers)"),
        ("Gating Mechanism", "Time-step-varying softmax tensor w: [B, H, 3] (sum to 1)"),
        ("Output Dimension", "[Batch, Horizon, 4 targets] (Physical commodity scale)"),
    ]
    y_pos = 4.8
    for label, val in hyb_specs:
        ax.text(8.7, y_pos, f"• {label}:", fontsize=8.8, fontweight="bold", color="#1e3a8a")
        ax.text(8.9, y_pos - 0.28, val, fontsize=8.0, color="#1e293b")
        y_pos -= 0.55

    save_fig(fig, FIG_DIR / "14_model_comparison.png")


if __name__ == "__main__":
    print("[INFO] Rendering GUMNet and Model Comparison 3D diagrams...")
    draw_8_gumnet_overview()
    draw_9_gumnet_wavelet_kan()
    draw_10_gumnet_dilated_cnn()
    draw_11_gumnet_gru_attention()
    draw_12_gumnet_adaptive_gating()
    draw_13_gumnet_quantile_output()
    draw_14_model_comparison()
    print("[SUCCESS] All GUMNet and Comparison diagrams generated successfully at 300 DPI.")

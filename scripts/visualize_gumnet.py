# -*- coding: utf-8 -*-
"""
High-fidelity Isometric / 3D Volumetric Tensor Block Visualization for GUMNet and Dual-Model Comparison.
Refined Typography & Spatial Layout:
- Expanded vertical & horizontal spacing to eliminate overlaps
- Automatic text wrapping (textwrap)
- Distinct regions for Title, Subtitle, Tensor Blocks, and Annotations
- Overview diagrams kept concise, detailed formulas isolated to specific branch figures
- Zero text collision or clipped labels
"""

import textwrap
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path

FIG_DIR = Path(__file__).resolve().parent.parent / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def draw_isometric_cube(ax, x, y, w, h, d, face_color, edge_color="#334155", alpha=1.0, zorder=2):
    """
    Renders an isometric volumetric cuboid with distinct lighting on front, top, and right faces.
    """
    dx = d * 0.7
    dy = d * 0.5
    c_rgb = np.array(plt.matplotlib.colors.to_rgb(face_color))
    c_top = tuple(np.clip(c_rgb * 1.25, 0, 1))
    c_side = tuple(np.clip(c_rgb * 0.75, 0, 1))
    c_front = tuple(c_rgb)

    # Front
    front = Polygon([(x, y), (x + w, y), (x + w, y + h), (x, y + h)], closed=True, facecolor=c_front, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=zorder)
    ax.add_patch(front)

    # Top
    top = Polygon([(x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)], closed=True, facecolor=c_top, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=zorder + 1)
    ax.add_patch(top)

    # Side
    side = Polygon([(x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)], closed=True, facecolor=c_side, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=zorder + 1)
    ax.add_patch(side)


def add_wrapped_text(ax, x, y, text, width=28, fontsize=9, color="#1e293b", fontweight="normal", ha="center", va="center", bbox=None, zorder=5):
    """
    Wraps long descriptions gracefully with consistent line spacing.
    """
    wrapped = "\n".join(textwrap.wrap(text, width=width))
    return ax.text(x, y, wrapped, ha=ha, va=va, fontsize=fontsize, fontweight=fontweight, color=color, linespacing=1.25, bbox=bbox, zorder=zorder)


def save_fig(fig, filepath):
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Rendered: {filepath}")


# =============================================================================
# 8. GUMNet Architecture Overview (Expanded Vertical Canvas: Y in [0, 14])
# =============================================================================
def draw_8_gumnet_overview():
    fig, ax = plt.subplots(figsize=(17, 13), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 13)
    ax.axis("off")

    # Header in dedicated zone (Y: 12.0 - 12.8)
    ax.text(8.5, 12.5, "8. GUMNet Architecture Overview (3D Volumetric Tensor Flow)", fontsize=16, fontweight="bold", color="#0f172a", ha="center")
    ax.text(8.5, 12.1, "Multi-Horizon Quantile Forecaster with Wavelet-KAN, Dilated CNN, and GRU-Attention", fontsize=11, color="#64748b", ha="center")

    # Input Tensor (X: 0.8 -> 2.6, Y: 5.0 -> 7.0)
    draw_isometric_cube(ax, 0.8, 5.2, 1.8, 2.2, 0.8, "#bae6fd", edge_color="#0284c7")
    ax.text(1.7, 8.2, "Input Tensor X\n[B, 30, 17]", fontsize=10.5, fontweight="bold", color="#0369a1", ha="center")
    ax.text(1.7, 4.4, "seq_len = 30 steps\ninput_dim = 17 features", fontsize=8.5, color="#334155", ha="center")

    # --- THREE SPACED PARALLEL BRANCHES ---
    # Branch 1: Dilated CNN (Top: Y = 9.2)
    draw_isometric_cube(ax, 4.6, 9.2, 2.8, 1.6, 0.6, "#fed7aa", edge_color="#ea580c")
    ax.text(6.0, 11.4, "Branch 1: Dilated CNN\n[B, 17, 30] -> [B, 64]", fontsize=10, fontweight="bold", color="#9a3412", ha="center")
    ax.text(6.0, 8.5, "Conv1d(k=3, pad=2, dil=2) + AvgPool1d\nFeature Vector f_cnn: [B, 64]", fontsize=8.2, color="#7c2d12", ha="center")

    # Branch 2: GRU + Attention (Middle: Y = 5.5)
    draw_isometric_cube(ax, 4.6, 5.5, 2.8, 1.6, 0.6, "#86efac", edge_color="#16a34a")
    ax.text(6.0, 7.7, "Branch 2: GRU + Attention\n[B, 30, 17] -> [B, 64]", fontsize=10, fontweight="bold", color="#14532d", ha="center")
    ax.text(6.0, 4.8, "2-layer GRU + 4-Head Attention\nFeature Vector f_gru: [B, 64]", fontsize=8.2, color="#166534", ha="center")

    # Branch 3: Wavelet-KAN (Bottom: Y = 1.8)
    draw_isometric_cube(ax, 4.6, 1.8, 2.8, 1.6, 0.6, "#fde047", edge_color="#ca8a04")
    ax.text(6.0, 4.0, "Branch 3: Wavelet-KAN\n[B, 30, 17] -> [B, 64]", fontsize=10, fontweight="bold", color="#854d0e", ha="center")
    ax.text(6.0, 1.1, "Mexican Hat Wavelet Basis + AvgPool1d\nFeature Vector f_kan: [B, 64]", fontsize=8.2, color="#713f12", ha="center")

    # Dynamic Gating Block (X: 9.3 -> 11.8, Y: 4.4 -> 7.8)
    draw_isometric_cube(ax, 9.3, 4.4, 2.6, 3.4, 0.8, "#d8b4fe", edge_color="#7e22ce")
    ax.text(10.6, 8.6, "Adaptive Softmax Gating\nConcat: [B, 192]", fontsize=10.5, fontweight="bold", color="#581c87", ha="center")
    gating_desc = "f_concat = [f_cnn, f_gru, f_kan]\nLinear(192, 32) -> ReLU -> Linear(32, 3)\nSoftmax weights w: [B, 3]\nw_cnn + w_gru + w_kan = 1.0\nf_fused in [B, 64]"
    ax.text(10.6, 5.9, gating_desc, fontsize=8.2, color="#3b0764", ha="center", linespacing=1.3)

    # Quantile Output Head (X: 13.5 -> 15.6, Y: 4.8 -> 7.4)
    draw_isometric_cube(ax, 13.5, 4.8, 2.2, 2.6, 0.8, "#fbcfe8", edge_color="#db2777")
    ax.text(14.6, 8.2, "Quantile Output Head\n[B, H, 4, 3]", fontsize=10.5, fontweight="bold", color="#9d174d", ha="center")
    ax.text(14.6, 4.0, "nn.Linear(64, H * 4 * 3)\nQuantiles: q = 0.10, 0.50, 0.90\n4 Refined Products\n[MG95, MG92, DOs]", fontsize=8.2, color="#831843", ha="center", linespacing=1.25)

    # Clean Non-Colliding Connective Arrows
    arrow_props = dict(arrowstyle="-|>", color="#475569", lw=2.0, mutation_scale=15)
    ax.annotate("", xy=(4.5, 10.0), xytext=(3.1, 6.7), arrowprops=arrow_props)
    ax.annotate("", xy=(4.5, 6.3), xytext=(3.1, 6.3), arrowprops=arrow_props)
    ax.annotate("", xy=(4.5, 2.6), xytext=(3.1, 5.9), arrowprops=arrow_props)

    arrow_purple = dict(arrowstyle="-|>", color="#7e22ce", lw=2.0, mutation_scale=15)
    ax.annotate("", xy=(9.2, 7.2), xytext=(7.8, 10.0), arrowprops=arrow_purple)
    ax.annotate("", xy=(9.2, 6.1), xytext=(7.8, 6.3), arrowprops=arrow_purple)
    ax.annotate("", xy=(9.2, 5.0), xytext=(7.8, 2.6), arrowprops=arrow_purple)

    ax.annotate("", xy=(13.4, 6.1), xytext=(12.3, 6.1), arrowprops=dict(arrowstyle="-|>", color="#db2777", lw=2.5, mutation_scale=18))

    save_fig(fig, FIG_DIR / "8_gumnet_overview.png")


# =============================================================================
# 9. GUMNet Wavelet-KAN (Separated Title, Subtitle & Isolated Cards)
# =============================================================================
def draw_9_gumnet_wavelet_kan():
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    # Dedicated Title Header Zone (Y: 6.6 - 7.3)
    ax.text(8.0, 7.1, "9. GUMNet Branch 1: Wavelet-KAN Architecture & Mexican Hat Basis", fontsize=15, fontweight="bold", color="#854d0e", ha="center")
    ax.text(8.0, 6.65, "Learnable translation and scale parameters continuous Mexican Hat wavelet transformation", fontsize=10.5, color="#713f12", ha="center")

    blocks = [
        {"x": 0.8, "y": 2.5, "w": 1.8, "h": 2.5, "d": 0.6, "color": "#bae6fd", "edge": "#0284c7", "title": "Sequence Input", "dim": "[B, 30, 17]", "desc": "seq_len = 30\ninput_dim = 17 features\nStandardized inputs"},
        {"x": 3.8, "y": 2.3, "w": 2.8, "h": 2.9, "d": 0.7, "color": "#fde047", "edge": "#ca8a04", "title": "Wavelet Normalization", "dim": "x_norm in [B, 30, 17]", "desc": "scale, trans in (1, 1, 17)\nx_norm = (x - trans) / (scale)\nLearnable parameter shift"},
        {"x": 7.8, "y": 2.1, "w": 3.4, "h": 3.3, "d": 0.8, "color": "#facc15", "edge": "#854d0e", "title": "Mexican Hat + Linear Base", "dim": "kan_out in [B, 30, 64]", "desc": "psi = (1 - x_norm^2) * exp(-0.5 * x_norm^2)\nbase = SiLU(Linear1(x))\nwav = Linear2(psi)\nout = base + wav"},
        {"x": 12.4, "y": 2.6, "w": 2.4, "h": 2.3, "d": 0.6, "color": "#fef08a", "edge": "#ca8a04", "title": "Adaptive AvgPool", "dim": "f_kan: [B, 64]", "desc": "Transpose (1, 2) -> (B, 64, 30)\nAdaptiveAvgPool1d(1)\nSqueeze -> [B, 64]"},
    ]

    for b in blocks:
        draw_isometric_cube(ax, b["x"], b["y"], b["w"], b["h"], b["d"], b["color"], edge_color=b["edge"])
        # Stage title above block with clean spacing
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.75, b["title"], fontsize=9.5, fontweight="bold", color="#854d0e", ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.38, b["dim"], fontsize=8.5, fontweight="bold", color=b["edge"], ha="center")
        # Subtitle and math description safely below block
        add_wrapped_text(ax, b["x"] + b["w"] / 2, b["y"] - 0.9, b["desc"], width=28, fontsize=8.0, color="#713f12", va="top")

    arrow_props = dict(arrowstyle="-|>", color="#ca8a04", lw=2.0, mutation_scale=15)
    for i in range(len(blocks) - 1):
        x1 = blocks[i]["x"] + blocks[i]["w"] + blocks[i]["d"] * 0.7 + 0.05
        x2 = blocks[i + 1]["x"] - 0.05
        y_mid = blocks[i]["y"] + blocks[i]["h"] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "9_gumnet_wavelet_kan.png")


# =============================================================================
# 10. GUMNet Dilated CNN (Expanded Spacing & Clean Geometry)
# =============================================================================
def draw_10_gumnet_dilated_cnn():
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(8.0, 7.1, "10. GUMNet Branch 2: Dilated 1D Convolution (Expanded Receptive Field)", fontsize=15, fontweight="bold", color="#9a3412", ha="center")
    ax.text(8.0, 6.65, "Dilated causal convolution captures local multiscale volatility patterns without parameter expansion", fontsize=10.5, color="#7c2d12", ha="center")

    blocks = [
        {"x": 0.8, "y": 2.5, "w": 1.8, "h": 2.5, "d": 0.6, "color": "#bae6fd", "edge": "#0284c7", "title": "Sequence Input", "dim": "[B, 30, 17]", "desc": "seq_len = 30\ninput_dim = 17 features\nInput time-series"},
        {"x": 3.8, "y": 2.6, "w": 2.2, "h": 2.3, "d": 0.6, "color": "#fed7aa", "edge": "#ea580c", "title": "Channel Transpose", "dim": "x_cnn: [B, 17, 30]", "desc": "Transpose dim 1 and 2\nChannels = 17, Length = 30\nPrepares 1D conv format"},
        {"x": 7.4, "y": 2.2, "w": 3.6, "h": 3.1, "d": 0.8, "color": "#fdba74", "edge": "#c2410c", "title": "Dilated Conv1d + ReLU", "dim": "conv_out: [B, 64, 30]", "desc": "Conv1d(in=17, out=64, k=3, pad=2, dil=2)\nReceptive Field = 5 steps\nExtracts local price spikes"},
        {"x": 12.3, "y": 2.6, "w": 2.4, "h": 2.3, "d": 0.6, "color": "#fed7aa", "edge": "#ea580c", "title": "Adaptive AvgPool", "dim": "f_cnn: [B, 64]", "desc": "AdaptiveAvgPool1d(1)\nSqueezes temporal dim\nOutput feature vector"},
    ]

    for b in blocks:
        draw_isometric_cube(ax, b["x"], b["y"], b["w"], b["h"], b["d"], b["color"], edge_color=b["edge"])
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.75, b["title"], fontsize=9.5, fontweight="bold", color="#9a3412", ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.38, b["dim"], fontsize=8.5, fontweight="bold", color=b["edge"], ha="center")
        add_wrapped_text(ax, b["x"] + b["w"] / 2, b["y"] - 0.9, b["desc"], width=28, fontsize=8.0, color="#7c2d12", va="top")

    arrow_props = dict(arrowstyle="-|>", color="#ea580c", lw=2.0, mutation_scale=15)
    for i in range(len(blocks) - 1):
        x1 = blocks[i]["x"] + blocks[i]["w"] + blocks[i]["d"] * 0.7 + 0.05
        x2 = blocks[i + 1]["x"] - 0.05
        y_mid = blocks[i]["y"] + blocks[i]["h"] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "10_gumnet_dilated_cnn.png")


# =============================================================================
# 11. GUMNet GRU + Multi-Head Attention (Spacious Layout)
# =============================================================================
def draw_11_gumnet_gru_attention():
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(8.0, 7.1, "11. GUMNet Branch 3: 2-Layer GRU + Multi-Head Self-Attention", fontsize=15, fontweight="bold", color="#14532d", ha="center")
    ax.text(8.0, 6.65, "Stacked recurrent gating followed by multi-head self-attention extracting cross-timestep temporal dependencies", fontsize=10.5, color="#166534", ha="center")

    blocks = [
        {"x": 0.8, "y": 2.5, "w": 1.8, "h": 2.5, "d": 0.6, "color": "#bae6fd", "edge": "#0284c7", "title": "Sequence Input", "dim": "[B, 30, 17]", "desc": "seq_len = 30\ninput_dim = 17 features\nInput time-series"},
        {"x": 3.8, "y": 2.2, "w": 2.8, "h": 3.1, "d": 0.7, "color": "#86efac", "edge": "#16a34a", "title": "2-Layer Recurrent GRU", "dim": "gru_out: [B, 30, 64]", "desc": "in=17, hidden=64, num_layers=2\ndropout=0.1, batch_first=True\nRecurrent state dynamics"},
        {"x": 7.9, "y": 2.2, "w": 3.2, "h": 3.1, "d": 0.8, "color": "#4ade80", "edge": "#15803d", "title": "Multi-Head Attention", "dim": "attn_out: [B, 30, 64]", "desc": "embed_dim=64, num_heads=4\nSelf-Attention (Q=K=V=gru_out)\nCross-step temporal context"},
        {"x": 12.3, "y": 2.6, "w": 2.4, "h": 2.3, "d": 0.6, "color": "#bbf7d0", "edge": "#14532d", "title": "Terminal Context", "dim": "f_gru: [B, 64]", "desc": "Extract last step:\nattn_out[:, -1, :]\nFinal temporal context"},
    ]

    for b in blocks:
        draw_isometric_cube(ax, b["x"], b["y"], b["w"], b["h"], b["d"], b["color"], edge_color=b["edge"])
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.75, b["title"], fontsize=9.5, fontweight="bold", color="#14532d", ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.38, b["dim"], fontsize=8.5, fontweight="bold", color=b["edge"], ha="center")
        add_wrapped_text(ax, b["x"] + b["w"] / 2, b["y"] - 0.9, b["desc"], width=28, fontsize=8.0, color="#052e16", va="top")

    arrow_props = dict(arrowstyle="-|>", color="#15803d", lw=2.0, mutation_scale=15)
    for i in range(len(blocks) - 1):
        x1 = blocks[i]["x"] + blocks[i]["w"] + blocks[i]["d"] * 0.7 + 0.05
        x2 = blocks[i + 1]["x"] - 0.05
        y_mid = blocks[i]["y"] + blocks[i]["h"] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "11_gumnet_gru_attention.png")


# =============================================================================
# 12. GUMNet Adaptive Softmax Gating (Wide Format)
# =============================================================================
def draw_12_gumnet_adaptive_gating():
    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(8.0, 8.0, "12. GUMNet Adaptive Softmax Gating Mechanism", fontsize=16, fontweight="bold", color="#581c87", ha="center")
    ax.text(8.0, 7.55, "Learned dynamic fusion balancing CNN edges, GRU temporal momentum, and Wavelet-KAN non-linearity", fontsize=10.5, color="#6b21a8", ha="center")

    # 3 Inputs (Y: 5.6, 3.6, 1.6)
    draw_isometric_cube(ax, 0.8, 5.6, 2.0, 1.3, 0.5, "#fed7aa", edge_color="#ea580c")
    ax.text(1.8, 7.2, "f_cnn: [B, 64]\nDilated Conv1D", fontsize=8.5, fontweight="bold", color="#9a3412", ha="center")

    draw_isometric_cube(ax, 0.8, 3.6, 2.0, 1.3, 0.5, "#86efac", edge_color="#16a34a")
    ax.text(1.8, 5.2, "f_gru: [B, 64]\nGRU + Attention", fontsize=8.5, fontweight="bold", color="#14532d", ha="center")

    draw_isometric_cube(ax, 0.8, 1.6, 2.0, 1.3, 0.5, "#fde047", edge_color="#ca8a04")
    ax.text(1.8, 3.2, "f_kan: [B, 64]\nWavelet-KAN", fontsize=8.5, fontweight="bold", color="#854d0e", ha="center")

    # Concat
    draw_isometric_cube(ax, 4.2, 3.0, 2.2, 2.8, 0.7, "#e9d5ff", edge_color="#a855f7")
    ax.text(5.3, 6.2, "Concat Features\n[B, 192]", fontsize=10, fontweight="bold", color="#581c87", ha="center")
    ax.text(5.3, 2.4, "f_concat =\n[f_cnn, f_gru, f_kan]", fontsize=8, color="#6b21a8", ha="center")

    # Gating MLP
    draw_isometric_cube(ax, 7.7, 2.8, 2.6, 3.2, 0.8, "#c084fc", edge_color="#7e22ce")
    ax.text(9.0, 6.4, "Gating MLP\n192 -> 32 -> 3", fontsize=10, fontweight="bold", color="#3b0764", ha="center")
    ax.text(9.0, 4.4, "Linear(192, 32)\nReLU()\nLinear(32, 3)\nSoftmax(dim=-1)", fontsize=8, color="#2e1065", ha="center")
    ax.text(9.0, 2.2, "Weights w: [B, 3]\nSum to 1.0", fontsize=9, fontweight="bold", color="#581c87", ha="center")

    # Fused Output
    draw_isometric_cube(ax, 11.6, 3.0, 3.6, 2.8, 0.8, "#d8b4fe", edge_color="#7e22ce")
    ax.text(13.4, 6.2, "Fused Representation\nf_fused in [B, 64]", fontsize=10.5, fontweight="bold", color="#581c87", ha="center")
    ax.text(13.4, 4.5, "f_fused = (w_cnn * f_cnn) +\n          (w_gru * f_gru) +\n          (w_kan * f_kan)", fontsize=8.5, color="#3b0764", ha="center")
    ax.text(13.4, 3.3, "Direct input to Quantile Output Head", fontsize=8.5, fontweight="bold", color="#9333ea", ha="center")

    # Arrows
    arrow_purple = dict(arrowstyle="-|>", color="#7e22ce", lw=2.0, mutation_scale=15)
    ax.annotate("", xy=(4.1, 4.8), xytext=(3.0, 6.2), arrowprops=arrow_purple)
    ax.annotate("", xy=(4.1, 4.4), xytext=(3.0, 4.2), arrowprops=arrow_purple)
    ax.annotate("", xy=(4.1, 4.0), xytext=(3.0, 2.2), arrowprops=arrow_purple)

    ax.annotate("", xy=(7.6, 4.4), xytext=(6.5, 4.4), arrowprops=arrow_purple)
    ax.annotate("", xy=(11.5, 4.4), xytext=(10.4, 4.4), arrowprops=arrow_purple)

    save_fig(fig, FIG_DIR / "12_gumnet_adaptive_gating.png")


# =============================================================================
# 13. GUMNet Quantile Output Head (Spacious & Clean Layout)
# =============================================================================
def draw_13_gumnet_quantile_output():
    fig, ax = plt.subplots(figsize=(16, 8.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8.5)
    ax.axis("off")

    ax.text(8.0, 8.0, "13. GUMNet Quantile Output Head & Uncertainty Intervals", fontsize=16, fontweight="bold", color="#9d174d", ha="center")
    ax.text(8.0, 7.55, "Multi-quantile regression (q = 0.10, 0.50, 0.90) generating calibrated 80% probabilistic uncertainty bounds", fontsize=10.5, color="#831843", ha="center")

    draw_isometric_cube(ax, 0.8, 3.4, 2.0, 2.4, 0.6, "#d8b4fe", edge_color="#7e22ce")
    ax.text(1.8, 6.2, "Fused Vector\nf_fused in [B, 64]", fontsize=9.5, fontweight="bold", color="#581c87", ha="center")
    ax.text(1.8, 2.8, "Output of Gating", fontsize=8, color="#3b0764", ha="center")

    draw_isometric_cube(ax, 4.2, 3.2, 2.6, 2.8, 0.7, "#fbcfe8", edge_color="#db2777")
    ax.text(5.5, 6.4, "Linear Projection\n64 -> H * 4 * 3", fontsize=10, fontweight="bold", color="#9d174d", ha="center")
    ax.text(5.5, 4.4, "nn.Linear(64, H*4*3)\nReshape to:\n[B, H, 4, 3]", fontsize=8.5, color="#831843", ha="center")

    q_cards = [
        {"y": 5.4, "q": "q = 0.90", "name": "Upper Bound (P90)", "desc": "90th percentile pessimistic forecast / peak upside price risk", "bg": "#fce7f3", "edge": "#be185d"},
        {"y": 3.4, "q": "q = 0.50", "name": "Median Forecast (P50)", "desc": "Point-forecast median / primary benchmark evaluation metric", "bg": "#fbcfe8", "edge": "#db2777"},
        {"y": 1.4, "q": "q = 0.10", "name": "Lower Bound (P10)", "desc": "10th percentile optimistic forecast / downside price floor risk", "bg": "#fdf2f8", "edge": "#9d174d"},
    ]

    for c in q_cards:
        draw_isometric_cube(ax, 8.4, c["y"], 6.8, 1.4, 0.5, c["bg"], edge_color=c["edge"])
        ax.text(9.4, c["y"] + 0.7, c["q"], fontsize=12, fontweight="bold", color=c["edge"], va="center")
        ax.text(10.8, c["y"] + 0.85, c["name"], fontsize=10, fontweight="bold", color="#0f172a", va="center")
        ax.text(10.8, c["y"] + 0.45, c["desc"], fontsize=8.2, color="#334155", va="center")

    ax.annotate("", xy=(4.1, 4.6), xytext=(3.0, 4.6), arrowprops=dict(arrowstyle="-|>", color="#db2777", lw=2.2, mutation_scale=15))
    ax.annotate("", xy=(8.3, 6.1), xytext=(7.0, 5.0), arrowprops=dict(arrowstyle="-|>", color="#be185d", lw=2.0, mutation_scale=14))
    ax.annotate("", xy=(8.3, 4.1), xytext=(7.0, 4.6), arrowprops=dict(arrowstyle="-|>", color="#db2777", lw=2.0, mutation_scale=14))
    ax.annotate("", xy=(8.3, 2.1), xytext=(7.0, 4.2), arrowprops=dict(arrowstyle="-|>", color="#9d174d", lw=2.0, mutation_scale=14))

    save_fig(fig, FIG_DIR / "13_gumnet_quantile_output.png")


# =============================================================================
# 14. Model Comparison Diagram
# =============================================================================
def draw_14_model_comparison():
    fig, ax = plt.subplots(figsize=(17, 12.0), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 17)
    ax.set_ylim(0, 12.0)
    ax.axis("off")

    ax.text(8.5, 11.4, "14. System Model Comparison: GUMNet vs. HybridTriNet", fontsize=16, fontweight="bold", color="#0f172a", ha="center")
    ax.text(8.5, 10.95, "Comparative Architectural Paradigm, Objective Function, Target Handling & Operational Roles", fontsize=11, color="#64748b", ha="center")

    # Column 1: GUMNet Panel
    ax.add_patch(patches.FancyBboxPatch((0.6, 0.5), 7.6, 10.0, boxstyle="round,pad=0.2", facecolor="#fdf2f8", edgecolor="#db2777", lw=2.2))
    ax.text(4.4, 10.1, "GUMNet (Production & Backtest Pipeline)", fontsize=13, fontweight="bold", color="#9d174d", ha="center")
    ax.text(4.4, 9.7, "Probabilistic Multi-Horizon Quantile Forecaster", fontsize=9.5, style="italic", color="#be185d", ha="center")

    draw_isometric_cube(ax, 1.2, 7.9, 1.6, 1.3, 0.4, "#fbcfe8", edge_color="#db2777")
    ax.text(2.0, 8.5, "Input\n[B, 30, 17]", fontsize=8.5, fontweight="bold", color="#9d174d", ha="center")

    draw_isometric_cube(ax, 3.6, 7.9, 2.2, 1.3, 0.4, "#f472b6", edge_color="#be185d")
    ax.text(4.7, 8.5, "Tri-Branch\nWavelet+CNN+GRU", fontsize=8.5, fontweight="bold", color="#831843", ha="center")

    draw_isometric_cube(ax, 6.4, 7.9, 1.4, 1.3, 0.4, "#db2777", edge_color="#9d174d")
    ax.text(7.1, 8.5, "Output\n[B, H, 4, 3]", fontsize=8.5, fontweight="bold", color="#ffffff", ha="center")

    gum_specs = [
        ("Operational Role", ["Primary operational pipeline model /", "Candidate promotion backtester"]),
        ("Forecasting Paradigm", ["Probabilistic Quantile Forecast", "(q = 0.10, 0.50, 0.90)"]),
        ("Objective Loss", ["Pinball Quantile Loss", "(Multi-quantile asymmetric penalty)"]),
        ("Lookback Window", ["seq_len = 30 trading steps"]),
        ("Input Features", ["17 market & exogenous features"]),
        ("Representation Branches", [
            "1. Wavelet-KAN (Mexican Hat continuous basis)",
            "2. Dilated 1D CNN (dilation=2, k=3)",
            "3. 2-Layer GRU (hidden=64) + Multi-Head Attention"
        ]),
        ("Gating Mechanism", ["Softmax vector w: [B, 3] fusing feature representations"]),
        ("Output Dimension", ["[Batch, Horizon, 4 targets, 3 quantiles]"]),
    ]
    cur_y = 7.1
    for label, lines in gum_specs:
        ax.text(1.0, cur_y, f"• {label}:", fontsize=8.8, fontweight="bold", color="#831843")
        cur_y -= 0.28
        for line in lines:
            ax.text(1.3, cur_y, line, fontsize=8.0, color="#1e293b")
            cur_y -= 0.24
        cur_y -= 0.14

    # Column 2: HybridTriNet Panel
    ax.add_patch(patches.FancyBboxPatch((8.8, 0.5), 7.6, 10.0, boxstyle="round,pad=0.2", facecolor="#eff6ff", edgecolor="#2563eb", lw=2.2))
    ax.text(12.6, 10.1, "HybridTriNet (Clean Hub Packaging Release)", fontsize=13, fontweight="bold", color="#1e40af", ha="center")
    ax.text(12.6, 9.7, "Deterministic Deep Multi-Horizon Point Forecaster", fontsize=9.5, style="italic", color="#2563eb", ha="center")

    draw_isometric_cube(ax, 9.4, 7.9, 1.6, 1.3, 0.4, "#bfdbfe", edge_color="#2563eb")
    ax.text(10.2, 8.5, "Input\n[B, 64, 15]", fontsize=8.5, fontweight="bold", color="#1e40af", ha="center")

    draw_isometric_cube(ax, 11.8, 7.9, 2.2, 1.3, 0.4, "#60a5fa", edge_color="#1d4ed8")
    ax.text(12.9, 8.5, "Tri-Branch\nKAN+GRU+PatchAttn", fontsize=8.5, fontweight="bold", color="#172554", ha="center")

    draw_isometric_cube(ax, 14.6, 7.9, 1.4, 1.3, 0.4, "#2563eb", edge_color="#1e3a8a")
    ax.text(15.3, 8.5, "Output\n[B, H, 4]", fontsize=8.5, fontweight="bold", color="#ffffff", ha="center")

    hyb_specs = [
        ("Operational Role", ["Stand-alone clean 15-feature release", "packaged for Hugging Face Hub"]),
        ("Forecasting Paradigm", ["Deterministic Point Forecast with", "horizon-specific scalers"]),
        ("Objective Loss", ["Smooth L1 Loss", "(Huber Loss variant for outlier robustness)"]),
        ("Lookback Window", ["K = 64 trading steps (Extended history)"]),
        ("Input Features", ["15 clean market features", "(synthetic temporal index purged)"]),
        ("Representation Branches", [
            "1. KAN (Learnable RBF M=8, 2x KANBlocks)",
            "2. Sequential GRU (hidden=128, terminal state)",
            "3. Patch Transformer (7 patches, 4 heads, 2 layers)"
        ]),
        ("Gating Mechanism", ["Time-step-varying softmax tensor", "w: [B, H, 3] (sum to 1)"]),
        ("Output Dimension", ["[Batch, Horizon, 4 targets]", "(Physical commodity scale)"]),
    ]
    cur_y = 7.1
    for label, lines in hyb_specs:
        ax.text(9.2, cur_y, f"• {label}:", fontsize=8.8, fontweight="bold", color="#1e3a8a")
        cur_y -= 0.28
        for line in lines:
            ax.text(9.5, cur_y, line, fontsize=8.0, color="#1e293b")
            cur_y -= 0.24
        cur_y -= 0.14

    save_fig(fig, FIG_DIR / "14_model_comparison.png")


if __name__ == "__main__":
    print("[INFO] Re-rendering GUMNet and Model Comparison with perfected typography and spacing...")
    draw_8_gumnet_overview()
    draw_9_gumnet_wavelet_kan()
    draw_10_gumnet_dilated_cnn()
    draw_11_gumnet_gru_attention()
    draw_12_gumnet_adaptive_gating()
    draw_13_gumnet_quantile_output()
    draw_14_model_comparison()
    print("[SUCCESS] All GUMNet and Comparison figures re-rendered cleanly at 300 DPI.")

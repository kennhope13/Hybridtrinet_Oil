# -*- coding: utf-8 -*-
"""
High-fidelity Isometric / 3D Volumetric Tensor Block Visualization for HybridTriNet.
Matches the architectural standard of Palm-Vein with:
- Volumetric 3D tensor blocks (front, top, side faces with shading)
- Exact tensor dimensions [B, K, D_in], [B, H, D_out], hidden layers
- Orthogonal branch decomposition (KAN, GRU, Patch Transformer, Gated Fusion)
- Multi-horizon output cards and end-to-end pipeline
- Programmatic rendering at 300 DPI directly into docs/figures/
"""

import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import numpy as np
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "benchmark" / "retrain_clean_15feat" / "hybrid_h60_meta" / "feature_cols.json"
FIG_DIR = ROOT_DIR / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Actual architecture hyperparameters verified from code
K = 64
H = 60
D_in = 15
D_out = 4
d_feat = 96
kan_M = 8
kan_depth = 2
gru_hidden = 128
attn_dmodel = 64
attn_heads = 4
attn_layers = 2
patch_len = 16
stride = 8

TARGET_NAMES = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]


def draw_isometric_cube(ax, x, y, w, h, d, face_color, edge_color="#334155", alpha=1.0, zorder=2):
    """
    Draws a 3D isometric/oblique volumetric cuboid representing a tensor block.
    (x, y): bottom-left corner of front face
    w: width
    h: height
    d: depth offset (isometric projection dx, dy = d*0.7, d*0.5)
    """
    dx = d * 0.7
    dy = d * 0.5

    # Compute lighter/darker shades for 3D lighting effect
    c_rgb = np.array(plt.matplotlib.colors.to_rgb(face_color))
    c_top = tuple(np.clip(c_rgb * 1.25, 0, 1))      # Highlight top
    c_side = tuple(np.clip(c_rgb * 0.75, 0, 1))     # Shadow side
    c_front = tuple(c_rgb)

    # 1. Front face
    front = Polygon([
        (x, y), (x + w, y), (x + w, y + h), (x, y + h)
    ], closed=True, facecolor=c_front, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=zorder)
    ax.add_patch(front)

    # 2. Top face
    top = Polygon([
        (x, y + h), (x + w, y + h), (x + w + dx, y + h + dy), (x + dx, y + h + dy)
    ], closed=True, facecolor=c_top, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=zorder + 1)
    ax.add_patch(top)

    # 3. Side face (Right)
    side = Polygon([
        (x + w, y), (x + w + dx, y + dy), (x + w + dx, y + h + dy), (x + w, y + h)
    ], closed=True, facecolor=c_side, edgecolor=edge_color, lw=1.2, alpha=alpha, zorder=zorder + 1)
    ax.add_patch(side)


def save_fig(fig, filepath):
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close()
    print(f"[SUCCESS] Rendered 3D diagram: {filepath}")


# =============================================================================
# 1. Pipeline Tổng Thể (3D Volumetric Pipeline)
# =============================================================================
def draw_1_forecasting_pipeline():
    fig, ax = plt.subplots(figsize=(16, 7), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Header
    ax.text(8.0, 6.6, "1. End-to-End Multi-Horizon Forecasting Pipeline", fontsize=16, fontweight="bold", color="#0f172a", ha="center")
    ax.text(8.0, 6.2, "Chronologically strictly causal tensor flow from raw financial feeds to denormalized predictions", fontsize=10.5, color="#64748b", ha="center")

    steps = [
        {"x": 0.6, "y": 2.2, "w": 1.4, "h": 2.5, "d": 0.5, "color": "#cbd5e1", "edge": "#475569", "title": "1. Ingestion", "t_shape": "(T, 15)", "desc": "Raw market series,\nUSD Index, GPR,\nCalendar features"},
        {"x": 2.7, "y": 2.2, "w": 1.4, "h": 2.5, "d": 0.5, "color": "#93c5fd", "edge": "#2563eb", "title": "2. Clean & Preproc", "t_shape": "(T, 15)", "desc": "Causal forward-fill,\nDrop missing dates,\nZero future lookahead"},
        {"x": 4.8, "y": 2.0, "w": 1.5, "h": 2.8, "d": 0.6, "color": "#fde047", "edge": "#ca8a04", "title": "3. Windowing", "t_shape": "[B, 64, 15]", "desc": "Lookback K=64\nTarget H in {1..60}\nChronological split"},
        {"x": 7.0, "y": 1.8, "w": 1.5, "h": 3.1, "d": 0.7, "color": "#c084fc", "edge": "#7e22ce", "title": "4. Scaling", "t_shape": "[B, 64, 15]", "desc": "(X - mu_x) / sd_x\nFit on Train split only\nHorizon-isolated"},
        {"x": 9.2, "y": 1.6, "w": 2.0, "h": 3.4, "d": 0.8, "color": "#86efac", "edge": "#16a34a", "title": "5. HybridTriNet", "t_shape": "[B, H*D_out]", "desc": "KAN + GRU +\nPatch Transformer\n+ Gated Softmax"},
        {"x": 12.0, "y": 2.0, "w": 1.6, "h": 2.8, "d": 0.6, "color": "#fdba74", "edge": "#ea580c", "title": "6. Denormalize", "t_shape": "[B, H, 4]", "desc": "out_norm * sd_y + mu_y\nReal Physical Units\n(USD / Barrel)"},
        {"x": 14.2, "y": 2.2, "w": 1.3, "h": 2.4, "d": 0.5, "color": "#f1f5f9", "edge": "#334155", "title": "7. Evaluation", "t_shape": "Metrics", "desc": "Holdout MAE/RMSE\nStreamlit Dashboard\nReport Generator"},
    ]

    for s in steps:
        draw_isometric_cube(ax, s["x"], s["y"], s["w"], s["h"], s["d"], s["color"], edge_color=s["edge"])
        # Top title & shape
        ax.text(s["x"] + s["w"] / 2, s["y"] + s["h"] + 0.65, s["title"], fontsize=9.5, fontweight="bold", color="#0f172a", ha="center")
        ax.text(s["x"] + s["w"] / 2, s["y"] + s["h"] + 0.35, s["t_shape"], fontsize=8.5, fontweight="bold", color=s["edge"], ha="center")
        # Bottom description
        ax.text(s["x"] + s["w"] / 2, s["y"] - 0.7, s["desc"], fontsize=7.5, color="#334155", ha="center", va="top")

    # Connective arrows
    arrow_props = dict(arrowstyle="-|>", color="#64748b", lw=2.0, mutation_scale=15)
    for i in range(len(steps) - 1):
        x1 = steps[i]["x"] + steps[i]["w"] + steps[i]["d"] * 0.7 + 0.05
        x2 = steps[i + 1]["x"] - 0.05
        y_mid = steps[i]["y"] + steps[i]["h"] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    # Bottom specifications banner
    ax.add_patch(patches.Rectangle((0.6, 0.3), 14.9, 0.6, facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.2, linestyle="--"))
    ax.text(8.0, 0.58, "Causal Invariance: Scalers and training weights are bound to past partitions; zero leakage of future mean/variance.", fontsize=8.5, color="#475569", ha="center")

    save_fig(fig, FIG_DIR / "1_forecasting_pipeline.png")


# =============================================================================
# 2. HybridTriNet Overview (3D Volumetric Tensor Flow)
# =============================================================================
def draw_2_hybridtrinet_overview():
    fig, ax = plt.subplots(figsize=(16, 9), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9)
    ax.axis("off")

    ax.text(8.0, 8.6, "2. HybridTriNet Neural Architecture (3D Volumetric Overview)", fontsize=16, fontweight="bold", color="#0f172a", ha="center")
    ax.text(8.0, 8.2, f"Tri-Branch Feature Decomposition & Gated Softmax Fusion | Input: [B, {K}, {D_in}] -> Output: [B, {H}, {D_out}]", fontsize=10.5, color="#64748b", ha="center")

    # 1. Input Tensor [B, 64, 15]
    draw_isometric_cube(ax, 0.8, 3.5, 1.8, 2.2, 0.8, "#93c5fd", edge_color="#1d4ed8")
    ax.text(1.7, 6.2, "Input Tensor X\n[B, 64, 15]", fontsize=10, fontweight="bold", color="#1e40af", ha="center")
    ax.text(1.7, 3.0, "K=64 Timesteps\nD_in=15 Clean Features", fontsize=8, color="#334155", ha="center")

    # Branch 1: KAN (Top)
    draw_isometric_cube(ax, 4.5, 5.8, 2.6, 1.4, 0.6, "#fde047", edge_color="#ca8a04")
    ax.text(5.8, 7.5, "Branch 1: KAN Block\n[B, 960] -> [B, 96]", fontsize=9.5, fontweight="bold", color="#854d0e", ha="center")
    ax.text(5.8, 5.3, "RBF Expansion (M=8) + 2x KANBlocks\nFeature Vector fk: [B, 96]", fontsize=7.8, color="#713f12", ha="center")

    # Branch 2: GRU (Middle)
    draw_isometric_cube(ax, 4.5, 3.6, 2.6, 1.4, 0.6, "#86efac", edge_color="#16a34a")
    ax.text(5.8, 5.2, "Branch 2: Sequential GRU\n[B, 64, 15] -> [B, 96]", fontsize=9.5, fontweight="bold", color="#14532d", ha="center")
    ax.text(5.8, 3.1, "128-d Recurrence + Terminal State\nFeature Vector fg: [B, 96]", fontsize=7.8, color="#166534", ha="center")

    # Branch 3: Patch Transformer (Bottom)
    draw_isometric_cube(ax, 4.5, 1.4, 2.6, 1.4, 0.6, "#cbd5e1", edge_color="#475569")
    ax.text(5.8, 2.9, "Branch 3: Patch Transformer\n[B, 7, 64] -> [B, 96]", fontsize=9.5, fontweight="bold", color="#0f172a", ha="center")
    ax.text(5.8, 0.9, "7 Patches (p=16, s=8) + 4-Head Attention\nFeature Vector fa: [B, 96]", fontsize=7.8, color="#334155", ha="center")

    # Independent Branch Heads (Pre-fusion forecasts)
    draw_isometric_cube(ax, 8.4, 6.0, 1.2, 1.0, 0.4, "#fef08a", edge_color="#ca8a04")
    ax.text(9.0, 7.2, "yk: [B, H, 4]", fontsize=8.5, fontweight="bold", color="#854d0e", ha="center")

    draw_isometric_cube(ax, 8.4, 3.8, 1.2, 1.0, 0.4, "#bbf7d0", edge_color="#16a34a")
    ax.text(9.0, 5.0, "yg: [B, H, 4]", fontsize=8.5, fontweight="bold", color="#14532d", ha="center")

    draw_isometric_cube(ax, 8.4, 1.6, 1.2, 1.0, 0.4, "#e2e8f0", edge_color="#475569")
    ax.text(9.0, 2.8, "ya: [B, H, 4]", fontsize=8.5, fontweight="bold", color="#0f172a", ha="center")

    # Dynamic Gated Fusion Block
    draw_isometric_cube(ax, 10.7, 2.8, 2.2, 3.0, 0.8, "#d8b4fe", edge_color="#7e22ce")
    ax.text(11.8, 6.3, "Dynamic Gated Fusion\nConcat: [B, 288]", fontsize=10, fontweight="bold", color="#581c87", ha="center")
    ax.text(11.8, 3.8, "Gating MLP (288 -> 192 -> H*3)\nSoftmax Weights w: [B, H, 3]\nw_k + w_g + w_a = 1.0\ny = w_k*yk + w_g*yg + w_a*ya", fontsize=7.8, color="#3b0764", ha="center")

    # Final Output Tensor
    draw_isometric_cube(ax, 13.9, 3.5, 1.4, 2.0, 0.6, "#fdba74", edge_color="#c2410c")
    ax.text(14.6, 5.9, "Final Forecast\n[B, H, 4]", fontsize=10.5, fontweight="bold", color="#9a3412", ha="center")
    ax.text(14.6, 2.9, "H Steps Ahead\n4 Refined Products", fontsize=8, color="#7c2d12", ha="center")

    # Arrows
    arrow_props = dict(arrowstyle="-|>", color="#475569", lw=2.0, mutation_scale=15)
    # Input to branches
    ax.annotate("", xy=(4.4, 6.5), xytext=(2.9, 4.8), arrowprops=arrow_props)
    ax.annotate("", xy=(4.4, 4.3), xytext=(2.9, 4.5), arrowprops=arrow_props)
    ax.annotate("", xy=(4.4, 2.1), xytext=(2.9, 4.2), arrowprops=arrow_props)

    # Branches to branch heads
    ax.annotate("", xy=(8.3, 6.5), xytext=(7.4, 6.5), arrowprops=arrow_props)
    ax.annotate("", xy=(8.3, 4.3), xytext=(7.4, 4.3), arrowprops=arrow_props)
    ax.annotate("", xy=(8.3, 2.1), xytext=(7.4, 2.1), arrowprops=arrow_props)

    # Branch heads & features into Gated Fusion
    arrow_purple = dict(arrowstyle="-|>", color="#7e22ce", lw=2.0, mutation_scale=15)
    ax.annotate("", xy=(10.6, 5.0), xytext=(9.8, 6.5), arrowprops=arrow_purple)
    ax.annotate("", xy=(10.6, 4.3), xytext=(9.8, 4.3), arrowprops=arrow_purple)
    ax.annotate("", xy=(10.6, 3.6), xytext=(9.8, 2.1), arrowprops=arrow_purple)

    # Fusion to Output
    ax.annotate("", xy=(13.8, 4.5), xytext=(13.2, 4.5), arrowprops=dict(arrowstyle="-|>", color="#ea580c", lw=2.5, mutation_scale=18))

    save_fig(fig, FIG_DIR / "2_hybridtrinet_overview.png")


# =============================================================================
# 3. KAN Branch (3D Volumetric Decomposition)
# =============================================================================
def draw_3_kan_branch():
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(7.5, 6.0, "3. Branch 1: KAN (Kolmogorov-Arnold Network) Volumetric Architecture", fontsize=15, fontweight="bold", color="#78350f", ha="center")
    ax.text(7.5, 5.65, "Learnable Radial Basis Function expansions replace fixed activations to capture non-linear market interactions", fontsize=10, color="#92400e", ha="center")

    kan_blocks = [
        {"x": 0.6, "y": 2.0, "w": 1.6, "h": 2.6, "d": 0.6, "color": "#fef08a", "edge": "#ca8a04", "title": "Flattened Input", "dim": "[B, 960]", "desc": "Reshaped from\n[B, K=64, D_in=15]"},
        {"x": 2.9, "y": 2.2, "w": 1.5, "h": 2.2, "d": 0.5, "color": "#fde047", "edge": "#b45309", "title": "Linear In", "dim": "[B, 15]", "desc": "nn.Linear(960, 15)\nDimensionality\ncompression"},
        {"x": 5.1, "y": 1.8, "w": 2.0, "h": 3.0, "d": 0.7, "color": "#f59e0b", "edge": "#78350f", "title": "RBF Expansion", "dim": "[B, 120]", "desc": "Centers c in [-1, 1], M=8\nphi = exp(-gamma*(x-c)^2)\nOutput: 15 * 8 = 120"},
        {"x": 7.8, "y": 1.6, "w": 2.4, "h": 3.4, "d": 0.8, "color": "#fbbf24", "edge": "#92400e", "title": "2x KANBlocks", "dim": "[B, 15]", "desc": "Linear(120, 96) + GELU\nDropout(0.2) + Linear(96, 15)\nResidual: z + Proj(x)"},
        {"x": 10.9, "y": 2.2, "w": 1.6, "h": 2.2, "d": 0.6, "color": "#fde047", "edge": "#b45309", "title": "KAN Head", "dim": "fk: [B, 96]", "desc": "LayerNorm(15)\nLinear(15, 96)\nFused feature"},
        {"x": 13.1, "y": 2.4, "w": 1.3, "h": 1.8, "d": 0.5, "color": "#fed7aa", "edge": "#c2410c", "title": "Branch Head", "dim": "yk: [B, H, 4]", "desc": "Linear(96, H*D_out)\nKAN sub-forecast"},
    ]

    for b in kan_blocks:
        draw_isometric_cube(ax, b["x"], b["y"], b["w"], b["h"], b["d"], b["color"], edge_color=b["edge"])
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.65, b["title"], fontsize=9.5, fontweight="bold", color="#78350f", ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.35, b["dim"], fontsize=8.5, fontweight="bold", color=b["edge"], ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] - 0.7, b["desc"], fontsize=7.5, color="#451a03", ha="center", va="top")

    arrow_props = dict(arrowstyle="-|>", color="#92400e", lw=2.0, mutation_scale=15)
    for i in range(len(kan_blocks) - 1):
        x1 = kan_blocks[i]["x"] + kan_blocks[i]["w"] + kan_blocks[i]["d"] * 0.7 + 0.05
        x2 = kan_blocks[i + 1]["x"] - 0.05
        y_mid = kan_blocks[i]["y"] + kan_blocks[i]["h"] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "3_kan_branch.png")


# =============================================================================
# 4. GRU Temporal Branch (3D Volumetric Decomposition)
# =============================================================================
def draw_4_gru_temporal_branch():
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(7.5, 6.0, "4. Branch 2: Sequential GRU (Temporal Momentum) Volumetric Architecture", fontsize=15, fontweight="bold", color="#14532d", ha="center")
    ax.text(7.5, 5.65, "Autoregressive recurrent gating for sequential dependency modeling and short-term volatility transitions", fontsize=10, color="#166534", ha="center")

    gru_blocks = [
        {"x": 0.8, "y": 2.0, "w": 1.8, "h": 2.6, "d": 0.7, "color": "#bbf7d0", "edge": "#16a34a", "title": "Sequential Input", "dim": "[B, 64, 15]", "desc": "64 sequential steps\n15 features per step\nBatch first = True"},
        {"x": 3.6, "y": 1.7, "w": 2.8, "h": 3.2, "d": 0.8, "color": "#86efac", "edge": "#15803d", "title": "GRU Recurrent Core", "dim": "[B, 64, 128]", "desc": "nn.GRU(in=15, hidden=128)\nnum_layers=1\nSequential hidden states"},
        {"x": 7.4, "y": 2.2, "w": 1.6, "h": 2.2, "d": 0.6, "color": "#4ade80", "edge": "#14532d", "title": "Terminal State", "dim": "h_64: [B, 128]", "desc": "Extract hg[:, -1, :]\nFinal temporal context\nSingle slice"},
        {"x": 9.9, "y": 2.2, "w": 1.7, "h": 2.2, "d": 0.6, "color": "#86efac", "edge": "#15803d", "title": "GRU Head", "dim": "fg: [B, 96]", "desc": "LayerNorm(128)\nLinear(128, 96)\nFused feature"},
        {"x": 12.5, "y": 2.4, "w": 1.4, "h": 1.8, "d": 0.5, "color": "#fed7aa", "edge": "#c2410c", "title": "Branch Head", "dim": "yg: [B, H, 4]", "desc": "Linear(96, H*D_out)\nGRU sub-forecast"},
    ]

    for b in gru_blocks:
        draw_isometric_cube(ax, b["x"], b["y"], b["w"], b["h"], b["d"], b["color"], edge_color=b["edge"])
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.65, b["title"], fontsize=9.5, fontweight="bold", color="#14532d", ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.35, b["dim"], fontsize=8.5, fontweight="bold", color=b["edge"], ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] - 0.7, b["desc"], fontsize=7.5, color="#052e16", ha="center", va="top")

    arrow_props = dict(arrowstyle="-|>", color="#15803d", lw=2.0, mutation_scale=15)
    for i in range(len(gru_blocks) - 1):
        x1 = gru_blocks[i]["x"] + gru_blocks[i]["w"] + gru_blocks[i]["d"] * 0.7 + 0.05
        x2 = gru_blocks[i + 1]["x"] - 0.05
        y_mid = gru_blocks[i]["y"] + gru_blocks[i]["h"] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "4_gru_temporal_branch.png")


# =============================================================================
# 5. Patch Transformer Branch (3D Volumetric Decomposition)
# =============================================================================
def draw_5_patch_transformer_branch():
    fig, ax = plt.subplots(figsize=(15, 6.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6.5)
    ax.axis("off")

    ax.text(7.5, 6.0, "5. Branch 3: Patch Transformer (Multi-Scale Attention) Volumetric Architecture", fontsize=15, fontweight="bold", color="#0f172a", ha="center")
    ax.text(7.5, 5.65, "Segments sequential observations into overlapping temporal patches with self-attention across long temporal spans", fontsize=10, color="#334155", ha="center")

    pt_blocks = [
        {"x": 0.6, "y": 2.0, "w": 1.6, "h": 2.6, "d": 0.6, "color": "#f1f5f9", "edge": "#64748b", "title": "Sequence Input", "dim": "[B, 64, 15]", "desc": "Lookback window\n64 trading steps"},
        {"x": 2.9, "y": 2.1, "w": 1.8, "h": 2.4, "d": 0.6, "color": "#e2e8f0", "edge": "#475569", "title": "Patchify Proj", "dim": "[B, 7, 64]", "desc": "p=16, s=8 -> 7 tokens\nLinear(16*15, 64)\nToken projection"},
        {"x": 5.4, "y": 2.1, "w": 1.8, "h": 2.4, "d": 0.6, "color": "#cbd5e1", "edge": "#334155", "title": "Pos Encoding", "dim": "[B, 7, 64]", "desc": "Sin/Cos PosEnc\nza = tokens + pe[:, :7, :]\nPreserves order"},
        {"x": 8.0, "y": 1.7, "w": 2.4, "h": 3.2, "d": 0.7, "color": "#94a3b8", "edge": "#1e293b", "title": "Transformer Core", "dim": "[B, 7, 64]", "desc": "2-Layer Encoder, 4 Heads\nd_model=64, FFN=256\nMulti-head Self-Attention"},
        {"x": 11.1, "y": 2.2, "w": 1.6, "h": 2.2, "d": 0.6, "color": "#cbd5e1", "edge": "#475569", "title": "Pool & Head", "dim": "fa: [B, 96]", "desc": "Mean pool (B, 64)\nLN + Linear(64, 96)\nFused feature"},
        {"x": 13.2, "y": 2.4, "w": 1.3, "h": 1.8, "d": 0.5, "color": "#fed7aa", "edge": "#c2410c", "title": "Branch Head", "dim": "ya: [B, H, 4]", "desc": "Linear(96, H*D_out)\nAttention forecast"},
    ]

    for b in pt_blocks:
        draw_isometric_cube(ax, b["x"], b["y"], b["w"], b["h"], b["d"], b["color"], edge_color=b["edge"])
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.65, b["title"], fontsize=9.5, fontweight="bold", color="#0f172a", ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] + b["h"] + 0.35, b["dim"], fontsize=8.5, fontweight="bold", color=b["edge"], ha="center")
        ax.text(b["x"] + b["w"] / 2, b["y"] - 0.7, b["desc"], fontsize=7.5, color="#1e293b", ha="center", va="top")

    arrow_props = dict(arrowstyle="-|>", color="#334155", lw=2.0, mutation_scale=15)
    for i in range(len(pt_blocks) - 1):
        x1 = pt_blocks[i]["x"] + pt_blocks[i]["w"] + pt_blocks[i]["d"] * 0.7 + 0.05
        x2 = pt_blocks[i + 1]["x"] - 0.05
        y_mid = pt_blocks[i]["y"] + pt_blocks[i]["h"] / 2
        ax.annotate("", xy=(x2, y_mid), xytext=(x1, y_mid), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "5_patch_transformer_branch.png")


# =============================================================================
# 6. Gated Fusion Mechanism (3D Volumetric Softmax Weights)
# =============================================================================
def draw_6_gated_fusion():
    fig, ax = plt.subplots(figsize=(15, 7.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(7.5, 7.0, "6. Dynamic Gated Fusion Mechanism (3D Volumetric Formulation)", fontsize=16, fontweight="bold", color="#581c87", ha="center")
    ax.text(7.5, 6.6, "Time-varying softmax gating enforces partition of unity (sum of weights = 1.0) at each forecast step", fontsize=10, color="#6b21a8", ha="center")

    # Three input tensors on left
    draw_isometric_cube(ax, 0.8, 4.8, 1.8, 1.3, 0.5, "#fde047", edge_color="#ca8a04")
    ax.text(1.7, 6.4, "fk: KAN [B, 96]\nyk: [B, H, 4]", fontsize=8.5, fontweight="bold", color="#854d0e", ha="center")

    draw_isometric_cube(ax, 0.8, 3.1, 1.8, 1.3, 0.5, "#86efac", edge_color="#16a34a")
    ax.text(1.7, 4.7, "fg: GRU [B, 96]\nyg: [B, H, 4]", fontsize=8.5, fontweight="bold", color="#14532d", ha="center")

    draw_isometric_cube(ax, 0.8, 1.4, 1.8, 1.3, 0.5, "#cbd5e1", edge_color="#475569")
    ax.text(1.7, 3.0, "fa: Attn [B, 96]\nya: [B, H, 4]", fontsize=8.5, fontweight="bold", color="#0f172a", ha="center")

    # Feature Concat Block [B, 288]
    draw_isometric_cube(ax, 3.9, 2.5, 2.0, 2.6, 0.7, "#e9d5ff", edge_color="#a855f7")
    ax.text(4.9, 5.5, "Concat [fk, fg, fa]\n[B, 288]", fontsize=10, fontweight="bold", color="#581c87", ha="center")
    ax.text(4.9, 2.0, "Joint representation\nTri-branch context", fontsize=8, color="#6b21a8", ha="center")

    # Gating Network MLP
    draw_isometric_cube(ax, 7.1, 2.3, 2.4, 3.0, 0.8, "#c084fc", edge_color="#7e22ce")
    ax.text(8.3, 5.7, "Gating MLP Core\n288 -> 192 -> H*3", fontsize=10, fontweight="bold", color="#3b0764", ha="center")
    ax.text(8.3, 3.8, "Linear(288, 192) + GELU\nDropout(0.1)\nLinear(192, H*3)\nSoftmax across 3 branches", fontsize=7.8, color="#2e1065", ha="center")
    ax.text(8.3, 1.8, "Gating Weights w\n[B, H, 3]", fontsize=9, fontweight="bold", color="#581c87", ha="center")

    # Weighted Combination Matrix
    draw_isometric_cube(ax, 10.6, 2.5, 3.8, 2.6, 0.8, "#fed7aa", edge_color="#ea580c")
    ax.text(12.5, 5.5, "Time-Varying Point Forecast\ny = w_k*yk + w_g*yg + w_a*ya", fontsize=10.5, fontweight="bold", color="#9a3412", ha="center")
    ax.text(12.5, 4.2, "Mathematical Guarantee:\nw_k(t) + w_g(t) + w_a(t) = 1.0 (Exact)\nEnforces convex dynamic combination\nOutput shape: [B, H, 4]", fontsize=8.5, color="#7c2d12", ha="center")
    ax.text(12.5, 3.0, "MG95, MG92, DO 0.001%, DO 0.05%", fontsize=9, fontweight="bold", color="#c2410c", ha="center")

    # Arrows
    arrow_purple = dict(arrowstyle="-|>", color="#7e22ce", lw=2.0, mutation_scale=15)
    ax.annotate("", xy=(3.8, 4.3), xytext=(2.8, 5.3), arrowprops=arrow_purple)
    ax.annotate("", xy=(3.8, 3.8), xytext=(2.8, 3.7), arrowprops=arrow_purple)
    ax.annotate("", xy=(3.8, 3.3), xytext=(2.8, 2.0), arrowprops=arrow_purple)

    ax.annotate("", xy=(7.0, 3.8), xytext=(6.1, 3.8), arrowprops=arrow_purple)
    ax.annotate("", xy=(10.5, 3.8), xytext=(9.7, 3.8), arrowprops=dict(arrowstyle="-|>", color="#ea580c", lw=2.5, mutation_scale=18))

    save_fig(fig, FIG_DIR / "6_gated_fusion.png")


# =============================================================================
# 7. Multi-Horizon Output Shapes (3D Volumetric Alignment)
# =============================================================================
def draw_7_multi_horizon_output():
    fig, ax = plt.subplots(figsize=(16, 7.5), dpi=300, facecolor="#ffffff")
    ax.set_facecolor("#ffffff")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.5)
    ax.axis("off")

    ax.text(8.0, 7.0, "7. Multi-Horizon Forecasting Output Tensor Decomposition", fontsize=16, fontweight="bold", color="#0f172a", ha="center")
    ax.text(8.0, 6.6, "Isolated, horizon-specialized checkpoints evaluated simultaneously for 4 target commodities", fontsize=10, color="#64748b", ha="center")

    hz_cards = [
        {"h": 1, "h_val": 1.2, "w_val": 1.4, "color": "#94a3b8", "edge": "#475569", "name": "Horizon 1d", "shape": "[B, 1, 4]", "cycle": "Spot / 1-Day Ahead\nFast Momentum"},
        {"h": 5, "h_val": 1.6, "w_val": 1.4, "color": "#60a5fa", "edge": "#2563eb", "name": "Horizon 5d", "shape": "[B, 5, 4]", "cycle": "1-Week Outlook\nInventory Cycle"},
        {"h": 10, "h_val": 2.0, "w_val": 1.4, "color": "#818cf8", "edge": "#4f46e5", "name": "Horizon 10d", "shape": "[B, 10, 4]", "cycle": "2-Week Forward\nRefinery Dispatch"},
        {"h": 15, "h_val": 2.3, "w_val": 1.4, "color": "#c084fc", "edge": "#7e22ce", "name": "Horizon 15d", "shape": "[B, 15, 4]", "cycle": "Mid-Month Tender\nArbitrage Spread"},
        {"h": 20, "h_val": 2.6, "w_val": 1.4, "color": "#f472b6", "edge": "#db2777", "name": "Horizon 20d", "shape": "[B, 20, 4]", "cycle": "Monthly Turnaround\nProcurement Cycle"},
        {"h": 30, "h_val": 2.9, "w_val": 1.4, "color": "#fbbf24", "edge": "#d97706", "name": "Horizon 30d", "shape": "[B, 30, 4]", "cycle": "1-Month Contract\nMacro Shifts"},
        {"h": 60, "h_val": 3.4, "w_val": 1.4, "color": "#fb923c", "edge": "#ea580c", "name": "Horizon 60d", "shape": "[B, 60, 4]", "cycle": "Quarterly Strategy\nStructural Cycle"},
    ]

    for idx, c in enumerate(hz_cards):
        x = 0.6 + idx * 2.15
        y = 2.0
        # Tensor box with height proportional to horizon length
        draw_isometric_cube(ax, x, y, c["w_val"], c["h_val"], 0.6, c["color"], edge_color=c["edge"])
        ax.text(x + c["w_val"] / 2, y + c["h_val"] + 0.65, c["name"], fontsize=10.5, fontweight="bold", color="#0f172a", ha="center")
        ax.text(x + c["w_val"] / 2, y + c["h_val"] + 0.35, c["shape"], fontsize=9, fontweight="bold", color=c["edge"], ha="center")
        ax.text(x + c["w_val"] / 2, y - 0.7, f"{c['cycle']}\nMG95, MG92\nDO 0.001%, DO 0.05%", fontsize=7.5, color="#334155", ha="center", va="top")

    # Bottom notice
    ax.add_patch(patches.Rectangle((0.6, 0.4), 14.8, 0.7, facecolor="#f8fafc", edgecolor="#cbd5e1", lw=1.2, linestyle="--"))
    ax.text(8.0, 0.75, "Denormalization Guarantee: Out = out_norm * y_sd + y_mu restoring forecasts to physical USD/barrel & market scale.", fontsize=8.8, color="#475569", ha="center")

    save_fig(fig, FIG_DIR / "7_multi_horizon_output.png")


if __name__ == "__main__":
    print("[INFO] Rendering all 7 volumetric 3D architectural diagrams...")
    draw_1_forecasting_pipeline()
    draw_2_hybridtrinet_overview()
    draw_3_kan_branch()
    draw_4_gru_temporal_branch()
    draw_5_patch_transformer_branch()
    draw_6_gated_fusion()
    draw_7_multi_horizon_output()
    print("[SUCCESS] All 7 diagrams rendered at 300 DPI.")

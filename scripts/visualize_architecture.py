import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "benchmark" / "retrain_clean_15feat" / "hybrid_h60_meta" / "feature_cols.json"
FIG_DIR = ROOT_DIR / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# Architecture Hyperparameters from real code
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

HORIZONS = [1, 5, 10, 15, 20, 30, 60]
TARGET_NAMES = ["MG95", "MG92", "DO 0.001%", "DO 0.05%"]


def save_fig(fig, filepath):
    plt.tight_layout()
    plt.savefig(filepath, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Generated: {filepath}")


# -------------------------------------------------------------
# 1. Pipeline Tổng Thể
# -------------------------------------------------------------
def draw_1_forecasting_pipeline():
    fig, ax = plt.subplots(figsize=(15, 6), dpi=300)
    ax.set_xlim(0, 15)
    ax.set_ylim(0, 6)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    ax.text(7.5, 5.5, "1. End-to-End Multi-Horizon Forecasting Pipeline", fontsize=15, fontweight="bold", ha="center", color="#0f172a")
    ax.text(7.5, 5.15, "Strict chronological causality: ingestion, causal scaling, neural forecasting & inverse scaling", fontsize=10, style="italic", ha="center", color="#475569")

    steps = [
        {"x": 0.5, "title": "Raw Data", "sub": "Petroleum prices,\nUSD Index, GPR,\nCalendar (2008-2026)", "bg": "#f8fafc", "border": "#64748b"},
        {"x": 2.1, "title": "Data Cleaning", "sub": "Drop missing dates,\nCausal ffill only,\nNo future lookahead", "bg": "#eff6ff", "border": "#3b82f6"},
        {"x": 3.7, "title": "15 Clean Features", "sub": "11 Exogenous/Time +\n4 Lagged Targets\n(Drop synthetic index)", "bg": "#e0e7ff", "border": "#4f46e5"},
        {"x": 5.3, "title": "Chrono Split", "sub": "Train (70%)\nVal (15%)\nHoldout (15%)\n2023-09-11 -> 2026", "bg": "#fef3c7", "border": "#d97706"},
        {"x": 6.9, "title": "Standardization", "sub": "Fit mu, sd on Train;\n(X - mu_x) / sd_x\nHorizon-specific", "bg": "#fae8ff", "border": "#a855f7"},
        {"x": 8.5, "title": "Windowing", "sub": "Lookback K=64\nTarget H in {1..60}\nTensor (B, 64, 15)", "bg": "#fce7f3", "border": "#ec4899"},
        {"x": 10.1, "title": "HybridTriNet", "sub": "KAN + GRU +\nPatch Transformer\n+ Dynamic Gate", "bg": "#dcfce7", "border": "#16a34a"},
        {"x": 11.7, "title": "Inverse Scaling", "sub": "out_norm * sd_y + mu_y\nPhysical units\n(USD / Barrel)", "bg": "#ffedd5", "border": "#ea580c"},
        {"x": 13.3, "title": "Holdout Eval", "sub": "MAE, MAPE, RMSE\nInteractive Dashboard\nPDF / CSV Export", "bg": "#f1f5f9", "border": "#334155"},
    ]

    for s in steps:
        box = patches.FancyBboxPatch((s["x"], 1.5), 1.35, 3.0, boxstyle="round,pad=0.1", facecolor=s["bg"], edgecolor=s["border"], linewidth=1.5)
        ax.add_patch(box)
        ax.text(s["x"] + 0.675, 4.0, s["title"], fontsize=9, fontweight="bold", ha="center", color="#0f172a")
        ax.text(s["x"] + 0.675, 2.7, s["sub"], fontsize=7.5, ha="center", color="#334155")

    arrow_props = dict(arrowstyle="-|>", color="#94a3b8", lw=1.5, mutation_scale=12)
    for i in range(len(steps) - 1):
        ax.annotate("", xy=(steps[i + 1]["x"], 3.0), xytext=(steps[i]["x"] + 1.35, 3.0), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "1_forecasting_pipeline.png")


# -------------------------------------------------------------
# 2. HybridTriNet Overview
# -------------------------------------------------------------
def draw_2_hybridtrinet_overview():
    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    ax.text(7.0, 7.5, "2. HybridTriNet Architecture Overview", fontsize=15, fontweight="bold", ha="center", color="#0f172a")
    ax.text(7.0, 7.15, "Tri-Branch Feature Extraction with Learned Dynamic Gating", fontsize=10, style="italic", ha="center", color="#64748b")

    # Input Box
    ax.add_patch(patches.FancyBboxPatch((0.5, 3.1), 2.2, 1.8, boxstyle="round,pad=0.2", facecolor="#e0e7ff", edgecolor="#4f46e5", linewidth=2))
    ax.text(1.6, 4.4, "Lookback Window", fontsize=11, fontweight="bold", ha="center", color="#312e81")
    ax.text(1.6, 3.9, f"Tensor: [B, {K}, {D_in}]", fontsize=10, ha="center", color="#4338ca")
    ax.text(1.6, 3.4, "K=64 Historical Steps\n15 Market Features", fontsize=8, ha="center", color="#6366f1")

    # 3 Branches
    ax.add_patch(patches.FancyBboxPatch((3.8, 5.2), 3.4, 1.6, boxstyle="round,pad=0.15", facecolor="#fef3c7", edgecolor="#d97706", linewidth=1.8))
    ax.text(5.5, 6.4, "Branch 1: KAN", fontsize=10.5, fontweight="bold", ha="center", color="#78350f")
    ax.text(5.5, 5.95, "Linear: (B, 64*15) -> (B, 15)", fontsize=8.5, ha="center", color="#92400e")
    ax.text(5.5, 5.55, "2x KANBlock (RBF M=8) -> fk: [B, 96]", fontsize=8.5, ha="center", color="#b45309")

    ax.add_patch(patches.FancyBboxPatch((3.8, 3.2), 3.4, 1.6, boxstyle="round,pad=0.15", facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.8))
    ax.text(5.5, 4.4, "Branch 2: Sequential GRU", fontsize=10.5, fontweight="bold", ha="center", color="#14532d")
    ax.text(5.5, 3.95, "GRU(in=15, hidden=128, layers=1)", fontsize=8.5, ha="center", color="#166534")
    ax.text(5.5, 3.55, "Head: LN + Linear -> fg: [B, 96]", fontsize=8.5, ha="center", color="#15803d")

    ax.add_patch(patches.FancyBboxPatch((3.8, 1.2), 3.4, 1.6, boxstyle="round,pad=0.15", facecolor="#f1f5f9", edgecolor="#475569", linewidth=1.8))
    ax.text(5.5, 2.4, "Branch 3: Patch Transformer", fontsize=10.5, fontweight="bold", ha="center", color="#0f172a")
    ax.text(5.5, 1.95, "Patchify(p=16, s=8) + PosEnc", fontsize=8.5, ha="center", color="#334155")
    ax.text(5.5, 1.55, "2-layer TransformerEncoder -> fa: [B, 96]", fontsize=8.5, ha="center", color="#475569")

    # Fusion
    ax.add_patch(patches.FancyBboxPatch((8.3, 2.4), 2.7, 3.2, boxstyle="round,pad=0.2", facecolor="#fae8ff", edgecolor="#a855f7", linewidth=2))
    ax.text(9.65, 5.1, "Dynamic Gated Fusion", fontsize=11, fontweight="bold", ha="center", color="#581c87")
    ax.text(9.65, 4.6, "Concat: [fk, fg, fa] -> [B, 288]", fontsize=8.5, ha="center", color="#7e22ce")
    ax.text(9.65, 4.0, "Gating MLP: 288 -> 192 -> H*3\nSoftmax across 3 branches", fontsize=8.5, ha="center", color="#6b21a8")
    ax.text(9.65, 3.3, "Weights w: [B, H, 3] (sum to 1)\ny = w_k*yk + w_g*yg + w_a*ya", fontsize=8.5, ha="center", color="#9333ea")

    # Output
    ax.add_patch(patches.FancyBboxPatch((11.9, 3.2), 1.9, 1.6, boxstyle="round,pad=0.2", facecolor="#ffedd5", edgecolor="#ea580c", linewidth=2))
    ax.text(12.85, 4.3, "Output Forecast", fontsize=10.5, fontweight="bold", ha="center", color="#7c2d12")
    ax.text(12.85, 3.8, f"Y: [B, {H}, {D_out}]", fontsize=11, fontweight="bold", ha="center", color="#c2410c")
    ax.text(12.85, 3.4, "MG95, MG92, DOs\n(Physical Scale)", fontsize=8, ha="center", color="#9a3412")

    # Arrows
    arrow_props = dict(arrowstyle="-|>", color="#64748b", lw=1.8, mutation_scale=15)
    ax.annotate("", xy=(3.8, 6.0), xytext=(2.7, 4.3), arrowprops=arrow_props)
    ax.annotate("", xy=(3.8, 4.0), xytext=(2.7, 4.0), arrowprops=arrow_props)
    ax.annotate("", xy=(3.8, 2.0), xytext=(2.7, 3.7), arrowprops=arrow_props)

    ax.annotate("", xy=(8.3, 4.8), xytext=(7.2, 6.0), arrowprops=arrow_props)
    ax.annotate("", xy=(8.3, 4.0), xytext=(7.2, 4.0), arrowprops=arrow_props)
    ax.annotate("", xy=(8.3, 3.2), xytext=(7.2, 2.0), arrowprops=arrow_props)

    ax.annotate("", xy=(11.9, 4.0), xytext=(11.0, 4.0), arrowprops=dict(arrowstyle="-|>", color="#a855f7", lw=2.2, mutation_scale=18))

    save_fig(fig, FIG_DIR / "2_hybridtrinet_overview.png")


# -------------------------------------------------------------
# 3. KAN Branch Details
# -------------------------------------------------------------
def draw_3_kan_branch():
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    ax.text(7.0, 5.0, "3. Branch 1: KAN (Kolmogorov-Arnold Network)", fontsize=14, fontweight="bold", ha="center", color="#78350f")
    ax.text(7.0, 4.65, "Non-linear feature interaction via learnable RBF basis expansions and residual blocks", fontsize=9.5, style="italic", ha="center", color="#92400e")

    blocks = [
        {"x": 0.5, "title": "Flattened Input", "sub": "X: [B, 64, 15]\nFlatten -> [B, 960]", "bg": "#fef3c7", "border": "#d97706"},
        {"x": 2.4, "title": "Linear In", "sub": "nn.Linear(960, 15)\nReduces dim -> [B, 15]", "bg": "#fffbeb", "border": "#b45309"},
        {"x": 4.3, "title": "RBF Expansion", "sub": "Centers c in [-1, 1], M=8\nphi = exp(-gamma*(x-c)^2)\nOutput: [B, 15*8 = 120]", "bg": "#fef3c7", "border": "#d97706"},
        {"x": 6.3, "title": "KAN Block (x2)", "sub": "Linear(120, 96) + GELU\nDropout(0.2) + Linear(96, 15)\nResidual: z + Proj(x)", "bg": "#fde68a", "border": "#b45309"},
        {"x": 8.7, "title": "KAN Head", "sub": "LayerNorm(15)\nLinear(15, 96)\nOutput fk: [B, 96]", "bg": "#fef3c7", "border": "#d97706"},
        {"x": 10.9, "title": "Branch Head Out", "sub": "nn.Linear(96, H*D_out)\nyk: [B, H, 4]\nIndependent forecast", "bg": "#fff7ed", "border": "#ea580c"},
    ]

    for b in blocks:
        box = patches.FancyBboxPatch((b["x"], 1.5), 1.6, 2.6, boxstyle="round,pad=0.15", facecolor=b["bg"], edgecolor=b["border"], linewidth=1.6)
        ax.add_patch(box)
        ax.text(b["x"] + 0.8, 3.6, b["title"], fontsize=9.5, fontweight="bold", ha="center", color="#78350f")
        ax.text(b["x"] + 0.8, 2.5, b["sub"], fontsize=8, ha="center", color="#451a03")

    arrow_props = dict(arrowstyle="-|>", color="#b45309", lw=1.6, mutation_scale=13)
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(blocks[i + 1]["x"], 2.8), xytext=(blocks[i]["x"] + 1.6, 2.8), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "3_kan_branch.png")


# -------------------------------------------------------------
# 4. GRU Temporal Branch Details
# -------------------------------------------------------------
def draw_4_gru_temporal_branch():
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    ax.text(7.0, 5.0, "4. Branch 2: Sequential GRU (Temporal Momentum)", fontsize=14, fontweight="bold", ha="center", color="#14532d")
    ax.text(7.0, 4.65, "Autoregressive recurrence capturing sequential dependencies across the 64-step horizon", fontsize=9.5, style="italic", ha="center", color="#166534")

    blocks = [
        {"x": 0.8, "title": "Sequence Input", "sub": "X: [B, 64, 15]\n64 sequential steps\n15 features per step", "bg": "#f0fdf4", "border": "#22c55e"},
        {"x": 3.2, "title": "Gated Recurrent Unit", "sub": "nn.GRU(D_in=15, hidden=128)\nnum_layers = 1\nbatch_first = True\nh_all: [B, 64, 128]", "bg": "#dcfce7", "border": "#16a34a"},
        {"x": 5.8, "title": "Last Hidden State", "sub": "h_last = hg[:, -1, :]\nState at step t=64\nShape: [B, 128]", "bg": "#bbf7d0", "border": "#15803d"},
        {"x": 8.3, "title": "GRU Head", "sub": "nn.LayerNorm(128)\nnn.Linear(128, 96)\nOutput fg: [B, 96]", "bg": "#dcfce7", "border": "#16a34a"},
        {"x": 10.8, "title": "Branch Head Out", "sub": "nn.Linear(96, H*D_out)\nyg: [B, H, 4]\nIndependent forecast", "bg": "#fff7ed", "border": "#ea580c"},
    ]

    for b in blocks:
        box = patches.FancyBboxPatch((b["x"], 1.5), 1.9, 2.6, boxstyle="round,pad=0.15", facecolor=b["bg"], edgecolor=b["border"], linewidth=1.6)
        ax.add_patch(box)
        ax.text(b["x"] + 0.95, 3.6, b["title"], fontsize=9.5, fontweight="bold", ha="center", color="#14532d")
        ax.text(b["x"] + 0.95, 2.5, b["sub"], fontsize=8, ha="center", color="#052e16")

    arrow_props = dict(arrowstyle="-|>", color="#16a34a", lw=1.6, mutation_scale=13)
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(blocks[i + 1]["x"], 2.8), xytext=(blocks[i]["x"] + 1.9, 2.8), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "4_gru_temporal_branch.png")


# -------------------------------------------------------------
# 5. Patch Transformer Branch Details
# -------------------------------------------------------------
def draw_5_patch_transformer_branch():
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    ax.text(7.0, 5.0, "5. Branch 3: Patch Transformer (Multi-Scale Attention)", fontsize=14, fontweight="bold", ha="center", color="#0f172a")
    ax.text(7.0, 4.65, "Tokenizes time-series into temporal patches with sinusoidal positional encodings and self-attention", fontsize=9.5, style="italic", ha="center", color="#334155")

    blocks = [
        {"x": 0.5, "title": "Sequence Input", "sub": "X: [B, 64, 15]\nLookback window\n64 trading steps", "bg": "#f8fafc", "border": "#64748b"},
        {"x": 2.4, "title": "Patchify Proj", "sub": "patch_len=16, stride=8\n7 patches per sample\nLinear(16*15, 64)\nTokens: [B, 7, 64]", "bg": "#f1f5f9", "border": "#475569"},
        {"x": 4.4, "title": "Positional Encoding", "sub": "Sin/Cos PosEnc\nFixed max_len=4096\nza = tokens + pe[:, :7, :]\nShape: [B, 7, 64]", "bg": "#e2e8f0", "border": "#334155"},
        {"x": 6.5, "title": "Transformer Encoder", "sub": "2 Layers, 4 Heads\nd_model=64, FFN=256\nDropout = 0.2\nOutput: [B, 7, 64]", "bg": "#cbd5e1", "border": "#1e293b"},
        {"x": 8.8, "title": "Global Pooling & Head", "sub": "Mean pool over patches: [B, 64]\nLayerNorm(64) + Linear(64, 96)\nOutput fa: [B, 96]", "bg": "#f1f5f9", "border": "#475569"},
        {"x": 11.1, "title": "Branch Head Out", "sub": "nn.Linear(96, H*D_out)\nya: [B, H, 4]\nIndependent forecast", "bg": "#fff7ed", "border": "#ea580c"},
    ]

    for b in blocks:
        box = patches.FancyBboxPatch((b["x"], 1.5), 1.65, 2.6, boxstyle="round,pad=0.15", facecolor=b["bg"], edgecolor=b["border"], linewidth=1.6)
        ax.add_patch(box)
        ax.text(b["x"] + 0.825, 3.6, b["title"], fontsize=9, fontweight="bold", ha="center", color="#0f172a")
        ax.text(b["x"] + 0.825, 2.5, b["sub"], fontsize=7.8, ha="center", color="#1e293b")

    arrow_props = dict(arrowstyle="-|>", color="#475569", lw=1.6, mutation_scale=13)
    for i in range(len(blocks) - 1):
        ax.annotate("", xy=(blocks[i + 1]["x"], 2.8), xytext=(blocks[i]["x"] + 1.65, 2.8), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "5_patch_transformer_branch.png")


# -------------------------------------------------------------
# 6. Gated Fusion Details
# -------------------------------------------------------------
def draw_6_gated_fusion():
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    ax.text(7.0, 6.0, "6. Dynamic Gated Fusion Mechanism", fontsize=15, fontweight="bold", ha="center", color="#581c87")
    ax.text(7.0, 5.65, "Learned softmax weighting enforces partition of unity (w_k + w_g + w_a = 1) per step", fontsize=9.5, style="italic", ha="center", color="#6b21a8")

    # Inputs from 3 branches
    ax.add_patch(patches.FancyBboxPatch((0.5, 4.2), 2.2, 1.0, boxstyle="round,pad=0.1", facecolor="#fef3c7", edgecolor="#d97706", linewidth=1.5))
    ax.text(1.6, 4.7, "fk: KAN Features [B, 96]\nyk: KAN Pred [B, H, 4]", fontsize=8.5, ha="center", color="#78350f")

    ax.add_patch(patches.FancyBboxPatch((0.5, 2.8), 2.2, 1.0, boxstyle="round,pad=0.1", facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.5))
    ax.text(1.6, 3.3, "fg: GRU Features [B, 96]\nyg: GRU Pred [B, H, 4]", fontsize=8.5, ha="center", color="#14532d")

    ax.add_patch(patches.FancyBboxPatch((0.5, 1.4), 2.2, 1.0, boxstyle="round,pad=0.1", facecolor="#f1f5f9", edgecolor="#475569", linewidth=1.5))
    ax.text(1.6, 1.9, "fa: Attn Features [B, 96]\nya: Attn Pred [B, H, 4]", fontsize=8.5, ha="center", color="#0f172a")

    # Concatenation Block
    ax.add_patch(patches.FancyBboxPatch((3.5, 2.6), 2.2, 1.6, boxstyle="round,pad=0.15", facecolor="#f3e8ff", edgecolor="#c084fc", linewidth=1.8))
    ax.text(4.6, 3.6, "Feature Concatenation", fontsize=9.5, fontweight="bold", ha="center", color="#581c87")
    ax.text(4.6, 3.1, "fcat = [fk, fg, fa]\nShape: [B, 288]", fontsize=8.5, ha="center", color="#6b21a8")

    # Gating MLP
    ax.add_patch(patches.FancyBboxPatch((6.5, 2.4), 2.8, 2.0, boxstyle="round,pad=0.15", facecolor="#fae8ff", edgecolor="#a855f7", linewidth=2))
    ax.text(7.9, 3.9, "Gating MLP + Softmax", fontsize=10, fontweight="bold", ha="center", color="#581c87")
    ax.text(7.9, 3.3, "Linear(288, 192) + GELU\nDropout(0.1)\nLinear(192, H*3)\nw = softmax(w, dim=-1)", fontsize=8, ha="center", color="#7e22ce")
    ax.text(7.9, 2.6, "Weights: w in [B, H, 3]", fontsize=8.5, fontweight="bold", ha="center", color="#9333ea")

    # Weighted Sum Formula
    ax.add_patch(patches.FancyBboxPatch((10.1, 2.3), 3.4, 2.2, boxstyle="round,pad=0.15", facecolor="#ffedd5", edgecolor="#ea580c", linewidth=2))
    ax.text(11.8, 4.0, "Time-Varying Fusion", fontsize=10.5, fontweight="bold", ha="center", color="#7c2d12")
    ax.text(11.8, 3.4, "y_t = w_k * yk_t + w_g * yg_t + w_a * ya_t\nw_k + w_g + w_a = 1.0 (Exact)", fontsize=8, ha="center", color="#9a3412")
    ax.text(11.8, 2.7, f"Final Forecast y: [B, {H}, {D_out}]", fontsize=10, fontweight="bold", ha="center", color="#c2410c")

    # Arrows
    arrow_props = dict(arrowstyle="-|>", color="#a855f7", lw=1.8, mutation_scale=14)
    ax.annotate("", xy=(3.5, 3.7), xytext=(2.7, 4.7), arrowprops=arrow_props)
    ax.annotate("", xy=(3.5, 3.4), xytext=(2.7, 3.3), arrowprops=arrow_props)
    ax.annotate("", xy=(3.5, 3.1), xytext=(2.7, 1.9), arrowprops=arrow_props)

    ax.annotate("", xy=(6.5, 3.4), xytext=(5.7, 3.4), arrowprops=arrow_props)
    ax.annotate("", xy=(10.1, 3.4), xytext=(9.3, 3.4), arrowprops=arrow_props)

    save_fig(fig, FIG_DIR / "6_gated_fusion.png")


# -------------------------------------------------------------
# 7. Multi-Horizon Output Shapes
# -------------------------------------------------------------
def draw_7_multi_horizon_output():
    fig, ax = plt.subplots(figsize=(14, 6.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 6.5)
    ax.axis("off")
    fig.patch.set_facecolor("#ffffff")

    ax.text(7.0, 6.0, "7. Multi-Horizon Forecasting Outputs & Target Alignment", fontsize=15, fontweight="bold", ha="center", color="#0f172a")
    ax.text(7.0, 5.65, "Independent model checkpoints trained and evaluated for each forecasting operational horizon", fontsize=9.5, style="italic", ha="center", color="#475569")

    # 7 Horizons cards
    hz_list = [
        {"h": 1, "shape": "[B, 1, 4]", "desc": "Next-day spot prediction\nHigh temporal momentum", "bg": "#f8fafc", "border": "#64748b"},
        {"h": 5, "shape": "[B, 5, 4]", "desc": "1-week trading outlook\nInventory cycle baseline", "bg": "#eff6ff", "border": "#3b82f6"},
        {"h": 10, "shape": "[B, 10, 4]", "desc": "2-week forward window\nBi-weekly delivery schedule", "bg": "#e0e7ff", "border": "#4f46e5"},
        {"h": 15, "shape": "[B, 15, 4]", "desc": "Mid-month procurement\nMacro trends & spread", "bg": "#fae8ff", "border": "#a855f7"},
        {"h": 20, "shape": "[B, 20, 4]", "desc": "1-month business cycle\nRefinery turnarounds", "bg": "#fce7f3", "border": "#ec4899"},
        {"h": 30, "shape": "[B, 30, 4]", "desc": "Monthly contract window\nMacro regime transitions", "bg": "#fef3c7", "border": "#d97706"},
        {"h": 60, "shape": "[B, 60, 4]", "desc": "Quarterly strategic planning\nLong-range cyclical shift", "bg": "#ffedd5", "border": "#ea580c"},
    ]

    for idx, hz in enumerate(hz_list):
        x = 0.5 + idx * 1.85
        box = patches.FancyBboxPatch((x, 1.8), 1.7, 3.4, boxstyle="round,pad=0.1", facecolor=hz["bg"], edgecolor=hz["border"], linewidth=1.6)
        ax.add_patch(box)
        ax.text(x + 0.85, 4.8, f"Horizon {hz['h']}d", fontsize=11, fontweight="bold", ha="center", color="#0f172a")
        ax.text(x + 0.85, 4.2, f"Shape:\n{hz['shape']}", fontsize=9.5, fontweight="bold", ha="center", color=hz["border"])
        ax.text(x + 0.85, 3.1, hz["desc"], fontsize=7.5, ha="center", color="#334155")
        ax.text(x + 0.85, 2.2, "4 Commodities:\nMG95, MG92,\nDO 0.001%, DO 0.05%", fontsize=6.8, ha="center", color="#64748b")

    # Bottom physical units reminder
    ax.add_patch(patches.FancyBboxPatch((0.5, 0.5), 13.0, 0.9, boxstyle="round,pad=0.1", facecolor="#f1f5f9", edgecolor="#cbd5e1", linewidth=1.2))
    ax.text(7.0, 0.95, "Outputs are natively inverse-standardized back to original physical currency/volume scale (USD per barrel / physical commodity units)", fontsize=9, style="italic", ha="center", color="#334155")

    save_fig(fig, FIG_DIR / "7_multi_horizon_output.png")


if __name__ == "__main__":
    draw_1_forecasting_pipeline()
    draw_2_hybridtrinet_overview()
    draw_3_kan_branch()
    draw_4_gru_temporal_branch()
    draw_5_patch_transformer_branch()
    draw_6_gated_fusion()
    draw_7_multi_horizon_output()

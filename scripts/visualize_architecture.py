import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
CONFIG_PATH = ROOT_DIR / "benchmark" / "retrain_clean_15feat" / "hybrid_h60_meta" / "feature_cols.json"
FIG_DIR = ROOT_DIR / "docs" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)


def draw_architecture():
    # Load config metadata
    K = 64
    H = 60
    D_in = 15
    D_out = 4
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH, "r", encoding="utf-8") as f:
            cfg = json.load(f)
            K = cfg.get("K", 64)
            H = cfg.get("H", 60)
            D_in = len(cfg.get("feature_cols", []))
            D_out = len(cfg.get("tgt_idx", []))

    fig, ax = plt.subplots(figsize=(14, 8), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Background styling
    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    # Title
    ax.text(
        7.0, 7.6,
        f"HybridTriNet Architecture (K={K}, H={H}, D_in={D_in}, D_out={D_out})",
        fontsize=16, fontweight="bold", ha="center", va="center", color="#1e293b"
    )
    ax.text(
        7.0, 7.2,
        "Tri-Branch Feature Extraction with Dynamic Gated Fusion",
        fontsize=11, style="italic", ha="center", va="center", color="#64748b"
    )

    # Input Block
    input_box = patches.FancyBboxPatch(
        (0.5, 3.2), 2.2, 1.6,
        boxstyle="round,pad=0.2", facecolor="#e0e7ff", edgecolor="#4f46e5", linewidth=2
    )
    ax.add_patch(input_box)
    ax.text(1.6, 4.3, "Lookback Input", fontsize=11, fontweight="bold", ha="center", va="center", color="#312e81")
    ax.text(1.6, 3.8, f"X: (B, {K}, {D_in})", fontsize=10, ha="center", va="center", color="#4338ca")
    ax.text(1.6, 3.4, "15 Market & Calendar Features", fontsize=8, ha="center", va="center", color="#6366f1")

    # Branch 1: KAN (Kolmogorov-Arnold Network)
    kan_box = patches.FancyBboxPatch(
        (3.8, 5.2), 3.2, 1.5,
        boxstyle="round,pad=0.2", facecolor="#fef3c7", edgecolor="#d97706", linewidth=1.5
    )
    ax.add_patch(kan_box)
    ax.text(5.4, 6.3, "Branch 1: KAN (Kolmogorov-Arnold)", fontsize=10, fontweight="bold", ha="center", color="#78350f")
    ax.text(5.4, 5.85, "Linear: (B, K*D_in) -> (B, D_in)", fontsize=8.5, ha="center", color="#92400e")
    ax.text(5.4, 5.5, "2x KANBlock (RBF M=8, hidden=96)", fontsize=8.5, ha="center", color="#b45309")

    # Branch 2: GRU
    gru_box = patches.FancyBboxPatch(
        (3.8, 3.25), 3.2, 1.5,
        boxstyle="round,pad=0.2", facecolor="#dcfce7", edgecolor="#16a34a", linewidth=1.5
    )
    ax.add_patch(gru_box)
    ax.text(5.4, 4.35, "Branch 2: Sequential GRU", fontsize=10, fontweight="bold", ha="center", color="#14532d")
    ax.text(5.4, 3.9, "GRU(D_in=15, hidden=128, layers=1)", fontsize=8.5, ha="center", color="#166534")
    ax.text(5.4, 3.55, "Head: LayerNorm + Linear -> (B, 96)", fontsize=8.5, ha="center", color="#15803d")

    # Branch 3: Patch Transformer
    attn_box = patches.FancyBboxPatch(
        (3.8, 1.3), 3.2, 1.5,
        boxstyle="round,pad=0.2", facecolor="#f1f5f9", edgecolor="#475569", linewidth=1.5
    )
    ax.add_patch(attn_box)
    ax.text(5.4, 2.4, "Branch 3: Patch Transformer", fontsize=10, fontweight="bold", ha="center", color="#0f172a")
    ax.text(5.4, 1.95, "Patchify(len=16, stride=8) + PosEnc", fontsize=8.5, ha="center", color="#334155")
    ax.text(5.4, 1.6, "TransformerEncoder (d_m=64, heads=4, layers=2)", fontsize=8, ha="center", color="#475569")

    # Dynamic Gated Fusion Block
    gate_box = patches.FancyBboxPatch(
        (8.0, 2.5), 2.5, 3.0,
        boxstyle="round,pad=0.2", facecolor="#fae8ff", edgecolor="#a855f7", linewidth=2
    )
    ax.add_patch(gate_box)
    ax.text(9.25, 5.0, "Dynamic Gated Fusion", fontsize=11, fontweight="bold", ha="center", color="#581c87")
    ax.text(9.25, 4.5, "Concat: [fk, fg, fa]\nShape: (B, 3 * 96)", fontsize=8.5, ha="center", color="#7e22ce")
    ax.text(9.25, 3.8, "Gate MLP: (288 -> 192 -> H*3)\nSoftmax across 3 branches", fontsize=8.5, ha="center", color="#6b21a8")
    ax.text(9.25, 3.1, "Weights: w = (B, H, 3)\nWeighted Combination", fontsize=8.5, ha="center", color="#9333ea")

    # Output Block
    out_box = patches.FancyBboxPatch(
        (11.4, 3.2), 2.1, 1.6,
        boxstyle="round,pad=0.2", facecolor="#ffedd5", edgecolor="#ea580c", linewidth=2
    )
    ax.add_patch(out_box)
    ax.text(12.45, 4.3, "Multi-Horizon Forecast", fontsize=10, fontweight="bold", ha="center", color="#7c2d12")
    ax.text(12.45, 3.8, f"Y: (B, {H}, {D_out})", fontsize=10.5, fontweight="bold", ha="center", color="#c2410c")
    ax.text(12.45, 3.4, "MG95, MG92, DOs\n(Original Scale)", fontsize=8, ha="center", color="#9a3412")

    # Connective Arrows
    arrow_props = dict(arrowstyle="-|>", color="#64748b", lw=1.8, mutation_scale=15)
    # Input -> Branches
    ax.annotate("", xy=(3.8, 5.95), xytext=(2.7, 4.2), arrowprops=arrow_props)
    ax.annotate("", xy=(3.8, 4.0), xytext=(2.7, 4.0), arrowprops=arrow_props)
    ax.annotate("", xy=(3.8, 2.05), xytext=(2.7, 3.8), arrowprops=arrow_props)

    # Branches -> Fusion
    ax.annotate("", xy=(8.0, 4.8), xytext=(7.0, 5.95), arrowprops=arrow_props)
    ax.annotate("", xy=(8.0, 4.0), xytext=(7.0, 4.0), arrowprops=arrow_props)
    ax.annotate("", xy=(8.0, 3.2), xytext=(7.0, 2.05), arrowprops=arrow_props)

    # Fusion -> Output
    ax.annotate("", xy=(11.4, 4.0), xytext=(10.5, 4.0), arrowprops=dict(arrowstyle="-|>", color="#a855f7", lw=2.2, mutation_scale=18))

    out_file = FIG_DIR / "hybridtrinet_architecture.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Saved architecture diagram: {out_file}")


def draw_pipeline():
    fig, ax = plt.subplots(figsize=(14, 5.5), dpi=300)
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 5.5)
    ax.axis("off")

    fig.patch.set_facecolor("#ffffff")
    ax.set_facecolor("#ffffff")

    ax.text(
        7.0, 5.0,
        "HybridTriNet End-to-End Time-Series Forecasting Pipeline",
        fontsize=15, fontweight="bold", ha="center", va="center", color="#1e293b"
    )
    ax.text(
        7.0, 4.65,
        "From Raw Financial/Energy Series to Denormalized Multi-Horizon Prediction",
        fontsize=10.5, style="italic", ha="center", va="center", color="#64748b"
    )

    steps = [
        {"x": 0.5, "title": "1. Ingestion", "desc": "Raw energy prices,\nUSD Index, GPR,\nCalendar features", "color": "#f1f5f9", "border": "#64748b"},
        {"x": 2.7, "title": "2. Preprocessing", "desc": "Causal ffill only,\nno future lookahead,\n15 clean features", "color": "#e0e7ff", "border": "#4f46e5"},
        {"x": 4.9, "title": "3. Windowing", "desc": "K=64 lookback,\nChronological split,\nStandardization (mu, sd)", "color": "#fef3c7", "border": "#d97706"},
        {"x": 7.1, "title": "4. Model Forward", "desc": "HybridTriNet (KAN+\nGRU+Patch Transformer\n+ Gated Fusion)", "color": "#fae8ff", "border": "#a855f7"},
        {"x": 9.3, "title": "5. Inverse Transform", "desc": "Denormalization\nout_norm * y_sd + y_mu\nOriginal target scale", "color": "#dcfce7", "border": "#16a34a"},
        {"x": 11.5, "title": "6. Downstream", "desc": "Holdout Evaluation,\nInteractive Dashboard,\nPDF Export / API", "color": "#ffedd5", "border": "#ea580c"},
    ]

    for s in steps:
        box = patches.FancyBboxPatch(
            (s["x"], 1.5), 1.95, 2.5,
            boxstyle="round,pad=0.15", facecolor=s["color"], edgecolor=s["border"], linewidth=1.6
        )
        ax.add_patch(box)
        ax.text(s["x"] + 0.97, 3.5, s["title"], fontsize=9.5, fontweight="bold", ha="center", va="center", color="#0f172a")
        ax.text(s["x"] + 0.97, 2.5, s["desc"], fontsize=8, ha="center", va="center", color="#334155")

    # Connective arrows between steps
    arrow_props = dict(arrowstyle="-|>", color="#94a3b8", lw=1.8, mutation_scale=14)
    for i in range(len(steps) - 1):
        x1 = steps[i]["x"] + 1.95
        x2 = steps[i + 1]["x"]
        ax.annotate("", xy=(x2, 2.75), xytext=(x1, 2.75), arrowprops=arrow_props)

    out_file = FIG_DIR / "forecasting_pipeline.png"
    plt.tight_layout()
    plt.savefig(out_file, dpi=300, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close()
    print(f"[SUCCESS] Saved pipeline diagram: {out_file}")


if __name__ == "__main__":
    draw_architecture()
    draw_pipeline()

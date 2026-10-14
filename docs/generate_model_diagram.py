import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_model_architecture_diagram():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Color palette (Clean Academic / Corporate Style)
    c_input = "#E2E8F0"
    c_kan = "#EDE9FE"       # Purple tint
    c_kan_border = "#8B5CF6"
    c_cnn = "#E0F2FE"       # Light blue
    c_cnn_border = "#0284C7"
    c_gru = "#DCFCE7"       # Green tint
    c_gru_border = "#16A34A"
    c_gate = "#FEF3C7"      # Amber
    c_gate_border = "#D97706"
    c_out = "#FEE2E2"       # Rose
    c_out_border = "#DC2626"
    c_box = "#F8FAFC"
    c_box_border = "#94A3B8"

    # Title
    ax.text(8, 9.5, "GUMNet: Multi-Branch Deep Learning Architecture with Adaptive Gating", 
            ha='center', va='center', fontsize=16, fontweight='bold', color="#0F172A")
    ax.text(8, 9.15, "Quantile Forecasting Engine with Wavelet-KAN, Dilated CNN, and GRU-Attention", 
            ha='center', va='center', fontsize=11, color="#475569")

    # 1. Input Block
    input_box = patches.FancyBboxPatch((0.5, 3.8), 2.2, 2.0, boxstyle="round,pad=0.15", 
                                       facecolor=c_input, edgecolor="#64748B", linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.6, 5.1, "Input Sequence", ha='center', va='center', fontsize=12, fontweight='bold', color="#1E293B")
    ax.text(1.6, 4.6, "[Batch, Seq_Len, Features]\n(Historical Prices +\nExogenous Signals)", 
            ha='center', va='center', fontsize=9, color="#475569")

    # Outer Container for Multi-Branch
    branch_container = patches.FancyBboxPatch((3.4, 1.4), 6.4, 7.0, boxstyle="round,pad=0.2", 
                                             facecolor=c_box, edgecolor=c_box_border, linewidth=1.5, linestyle="--")
    ax.add_patch(branch_container)
    ax.text(6.6, 8.1, "Multi-Branch Feature Extraction Ensembles", ha='center', va='center', 
            fontsize=12, fontweight='bold', color="#334155")

    # Branch 1: Wavelet-KAN
    b1 = patches.FancyBboxPatch((3.8, 6.0), 5.6, 1.6, boxstyle="round,pad=0.1", 
                                facecolor=c_kan, edgecolor=c_kan_border, linewidth=2)
    ax.add_patch(b1)
    ax.text(6.6, 7.1, "Branch 1: Wavelet-KAN Block", ha='center', va='center', fontsize=11, fontweight='bold', color="#4C1D95")
    ax.text(6.6, 6.5, "Mexican Hat Wavelet Basis: (1 - x²) · exp(-0.5·x²)\nSiLU Linear Base + Adaptive Average Pooling", 
            ha='center', va='center', fontsize=9, color="#5B21B6")

    # Branch 2: Dilated CNN
    b2 = patches.FancyBboxPatch((3.8, 4.0), 5.6, 1.6, boxstyle="round,pad=0.1", 
                                facecolor=c_cnn, edgecolor=c_cnn_border, linewidth=2)
    ax.add_patch(b2)
    ax.text(6.6, 5.1, "Branch 2: Dilated Conv1D Network", ha='center', va='center', fontsize=11, fontweight='bold', color="#0369A1")
    ax.text(6.6, 4.5, "Conv1D (Kernel=3, Dilation=2) + ReLU\nMulti-Scale Local Receptive Field + Adaptive Pooling", 
            ha='center', va='center', fontsize=9, color="#075985")

    # Branch 3: GRU + Multi-Head Attention
    b3 = patches.FancyBboxPatch((3.8, 2.0), 5.6, 1.6, boxstyle="round,pad=0.1", 
                                facecolor=c_gru, edgecolor=c_gru_border, linewidth=2)
    ax.add_patch(b3)
    ax.text(6.6, 3.1, "Branch 3: GRU + Multi-Head Attention", ha='center', va='center', fontsize=11, fontweight='bold', color="#15803D")
    ax.text(6.6, 2.5, "2-Layer Recurrent GRU (Dropout=0.1)\nMulti-Head Self-Attention (4 Heads) over Hidden States", 
            ha='center', va='center', fontsize=9, color="#166534")

    # Connecting arrows from Input to Branches
    arrow_props = dict(arrowstyle="->", color="#475569", lw=1.8, connectionstyle="arc3,rad=0")
    ax.annotate("", xy=(3.8, 6.8), xytext=(2.7, 5.2), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8, connectionstyle="angle,angleA=0,angleB=90,rad=5"))
    ax.annotate("", xy=(3.8, 4.8), xytext=(2.7, 4.8), arrowprops=arrow_props)
    ax.annotate("", xy=(3.8, 2.8), xytext=(2.7, 4.4), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8, connectionstyle="angle,angleA=0,angleB=90,rad=5"))

    # Feature representations
    ax.text(10.15, 6.8, "f_kan", fontsize=10, fontweight='bold', color="#5B21B6")
    ax.text(10.15, 4.8, "f_cnn", fontsize=10, fontweight='bold', color="#0369A1")
    ax.text(10.15, 2.8, "f_gru", fontsize=10, fontweight='bold', color="#15803D")

    # 3. Softmax Gating Block & Fusion
    gate_box = patches.FancyBboxPatch((10.7, 5.3), 2.2, 1.8, boxstyle="round,pad=0.1", 
                                      facecolor=c_gate, edgecolor=c_gate_border, linewidth=2)
    ax.add_patch(gate_box)
    ax.text(11.8, 6.5, "Softmax Gating", ha='center', va='center', fontsize=11, fontweight='bold', color="#92400E")
    ax.text(11.8, 5.9, "Concat [f_kan, f_cnn, f_gru]\nLinear(3*d) -> GELU -> Softmax\nw_kan, w_cnn, w_gru", 
            ha='center', va='center', fontsize=8.5, color="#78350F")

    fusion_circle = patches.Circle((11.8, 3.4), 0.65, facecolor="#F1F5F9", edgecolor="#334155", linewidth=2)
    ax.add_patch(fusion_circle)
    ax.text(11.8, 3.4, "Adaptive\nFusion", ha='center', va='center', fontsize=9, fontweight='bold', color="#1E293B")

    # Connect branches to Gate
    ax.annotate("", xy=(10.7, 6.5), xytext=(9.4, 6.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, connectionstyle="arc3,rad=-0.1"))
    ax.annotate("", xy=(10.7, 6.1), xytext=(9.4, 4.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, connectionstyle="arc3,rad=-0.1"))
    ax.annotate("", xy=(10.7, 5.7), xytext=(9.4, 2.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, connectionstyle="arc3,rad=0.1"))

    # Connect branches to Fusion
    ax.annotate("", xy=(11.2, 3.6), xytext=(9.4, 6.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, linestyle=":", connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(11.2, 3.4), xytext=(9.4, 4.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, linestyle=":", connectionstyle="arc3,rad=0.1"))
    ax.annotate("", xy=(11.2, 3.2), xytext=(9.4, 2.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, linestyle=":", connectionstyle="arc3,rad=-0.1"))

    # Connect Gate to Fusion
    ax.annotate("", xy=(11.8, 4.05), xytext=(11.8, 5.3), 
                arrowprops=dict(arrowstyle="->", color=c_gate_border, lw=2, linestyle="-"))
    ax.text(12.3, 4.65, "Weights (w)", fontsize=9, color="#92400E", fontweight='bold')

    # 4. Quantile Output Block
    out_box = patches.FancyBboxPatch((13.4, 2.4), 2.2, 2.0, boxstyle="round,pad=0.15", 
                                     facecolor=c_out, edgecolor=c_out_border, linewidth=2)
    ax.add_patch(out_box)
    ax.text(14.5, 3.8, "Quantile Head", ha='center', va='center', fontsize=12, fontweight='bold', color="#991B1B")
    ax.text(14.5, 3.1, "[Batch, Horizon, Quantiles]\n• q = 0.10 (Lower Bound)\n• q = 0.50 (Median Target)\n• q = 0.90 (Upper Bound)", 
            ha='center', va='center', fontsize=8.5, color="#7F1D1D")

    # Connect Fusion to Output
    ax.annotate("", xy=(13.4, 3.4), xytext=(12.45, 3.4), arrowprops=dict(arrowstyle="->", color="#0F172A", lw=2))

    # Save diagram
    out_dir = r"D:\user\projects\du_doan_gia_dau\Hybridtrinet_Oil\docs\images"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "gumnet_architecture.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    create_model_architecture_diagram()

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_hybrid_trinet_diagram():
    fig, ax = plt.subplots(figsize=(16, 10), dpi=300)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Palette
    c_input = "#E2E8F0"
    c_kan = "#EDE9FE"
    c_kan_border = "#8B5CF6"
    c_att = "#FEF08A"       # Warm yellow
    c_att_border = "#CA8A04"
    c_gru = "#DCFCE7"
    c_gru_border = "#16A34A"
    c_gate = "#FEF3C7"
    c_gate_border = "#D97706"
    c_out = "#FEE2E2"
    c_out_border = "#DC2626"
    c_box = "#F8FAFC"
    c_box_border = "#94A3B8"

    # Title
    ax.text(8, 9.5, "HybridTriNet: Tri-Branch Deep Forecasting Architecture", 
            ha='center', va='center', fontsize=16, fontweight='bold', color="#0F172A")
    ax.text(8, 9.15, "Integrating RBF-KAN, Patch-based Transformer Encoder, and GRU with Learnable Gating", 
            ha='center', va='center', fontsize=11, color="#475569")

    # 1. Input Block
    input_box = patches.FancyBboxPatch((0.5, 3.8), 2.2, 2.0, boxstyle="round,pad=0.15", 
                                       facecolor=c_input, edgecolor="#64748B", linewidth=2)
    ax.add_patch(input_box)
    ax.text(1.6, 5.1, "Input Sequence", ha='center', va='center', fontsize=12, fontweight='bold', color="#1E293B")
    ax.text(1.6, 4.6, "[Batch, Seq_Len=64, D_in]\n(Crude Oil Price +\nExogenous Features)", 
            ha='center', va='center', fontsize=9, color="#475569")

    # Outer Container
    branch_container = patches.FancyBboxPatch((3.4, 1.4), 6.4, 7.0, boxstyle="round,pad=0.2", 
                                             facecolor=c_box, edgecolor=c_box_border, linewidth=1.5, linestyle="--")
    ax.add_patch(branch_container)
    ax.text(6.6, 8.1, "Tri-Branch Heterogeneous Feature Ensembles", ha='center', va='center', 
            fontsize=12, fontweight='bold', color="#334155")

    # Branch 1: RBF-KAN
    b1 = patches.FancyBboxPatch((3.8, 6.0), 5.6, 1.6, boxstyle="round,pad=0.1", 
                                facecolor=c_kan, edgecolor=c_kan_border, linewidth=2)
    ax.add_patch(b1)
    ax.text(6.6, 7.1, "Branch 1: RBF-KAN Blocks", ha='center', va='center', fontsize=11, fontweight='bold', color="#4C1D95")
    ax.text(6.6, 6.5, "RBF Expansion (M=8 Gaussian Centers) -> GELU -> Linear\nResidual Projection + LayerNorm (d_feat=96)", 
            ha='center', va='center', fontsize=9, color="#5B21B6")

    # Branch 2: Patch Transformer
    b2 = patches.FancyBboxPatch((3.8, 4.0), 5.6, 1.6, boxstyle="round,pad=0.1", 
                                facecolor=c_att, edgecolor=c_att_border, linewidth=2)
    ax.add_patch(b2)
    ax.text(6.6, 5.1, "Branch 2: Patch Transformer Encoder", ha='center', va='center', fontsize=11, fontweight='bold', color="#854D0E")
    ax.text(6.6, 4.5, "Patchify (Len=16, Stride=8) + Sinusoidal Positional Encoding\nTransformerEncoder (Multi-Head Self-Attention, nhead=3)", 
            ha='center', va='center', fontsize=9, color="#713F12")

    # Branch 3: GRU
    b3 = patches.FancyBboxPatch((3.8, 2.0), 5.6, 1.6, boxstyle="round,pad=0.1", 
                                facecolor=c_gru, edgecolor=c_gru_border, linewidth=2)
    ax.add_patch(b3)
    ax.text(6.6, 3.1, "Branch 3: Sequential GRU", ha='center', va='center', fontsize=11, fontweight='bold', color="#15803D")
    ax.text(6.6, 2.5, "Gated Recurrent Unit (hidden_dim=128) -> LayerNorm\nCapturing short & mid-term temporal dynamics", 
            ha='center', va='center', fontsize=9, color="#166534")

    # Connecting arrows
    arrow_props = dict(arrowstyle="->", color="#475569", lw=1.8, connectionstyle="arc3,rad=0")
    ax.annotate("", xy=(3.8, 6.8), xytext=(2.7, 5.2), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8, connectionstyle="angle,angleA=0,angleB=90,rad=5"))
    ax.annotate("", xy=(3.8, 4.8), xytext=(2.7, 4.8), arrowprops=arrow_props)
    ax.annotate("", xy=(3.8, 2.8), xytext=(2.7, 4.4), arrowprops=dict(arrowstyle="->", color="#475569", lw=1.8, connectionstyle="angle,angleA=0,angleB=90,rad=5"))

    # Feature labels
    ax.text(10.15, 6.8, "fk (KAN)", fontsize=10, fontweight='bold', color="#5B21B6")
    ax.text(10.15, 4.8, "fa (Attn)", fontsize=10, fontweight='bold', color="#854D0E")
    ax.text(10.15, 2.8, "fg (GRU)", fontsize=10, fontweight='bold', color="#15803D")

    # 3. Gating & Dynamic Softmax Fusion
    gate_box = patches.FancyBboxPatch((10.7, 5.3), 2.2, 1.8, boxstyle="round,pad=0.1", 
                                      facecolor=c_gate, edgecolor=c_gate_border, linewidth=2)
    ax.add_patch(gate_box)
    ax.text(11.8, 6.5, "Adaptive Gating", ha='center', va='center', fontsize=11, fontweight='bold', color="#92400E")
    ax.text(11.8, 5.9, "Concat [fk, fg, fa]\nLinear(3*d) -> GELU -> Dropout\nLinear(H*3) -> Softmax", 
            ha='center', va='center', fontsize=8.5, color="#78350F")

    fusion_circle = patches.Circle((11.8, 3.4), 0.65, facecolor="#F1F5F9", edgecolor="#334155", linewidth=2)
    ax.add_patch(fusion_circle)
    ax.text(11.8, 3.4, "Multi-Horizon\nFusion", ha='center', va='center', fontsize=9, fontweight='bold', color="#1E293B")

    # Connect to Gate
    ax.annotate("", xy=(10.7, 6.5), xytext=(9.4, 6.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, connectionstyle="arc3,rad=-0.1"))
    ax.annotate("", xy=(10.7, 6.1), xytext=(9.4, 4.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, connectionstyle="arc3,rad=-0.1"))
    ax.annotate("", xy=(10.7, 5.7), xytext=(9.4, 2.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, connectionstyle="arc3,rad=0.1"))

    # Connect to Fusion
    ax.annotate("", xy=(11.2, 3.6), xytext=(9.4, 6.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, linestyle=":", connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(11.2, 3.4), xytext=(9.4, 4.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, linestyle=":", connectionstyle="arc3,rad=0.1"))
    ax.annotate("", xy=(11.2, 3.2), xytext=(9.4, 2.8), arrowprops=dict(arrowstyle="->", color="#64748B", lw=1.5, linestyle=":", connectionstyle="arc3,rad=-0.1"))

    # Connect Gate to Fusion
    ax.annotate("", xy=(11.8, 4.05), xytext=(11.8, 5.3), 
                arrowprops=dict(arrowstyle="->", color=c_gate_border, lw=2, linestyle="-"))
    ax.text(12.3, 4.65, "Weights: w(H, 3)", fontsize=9, color="#92400E", fontweight='bold')

    # 4. Multi-Horizon Output Block
    out_box = patches.FancyBboxPatch((13.4, 2.4), 2.2, 2.0, boxstyle="round,pad=0.15", 
                                     facecolor=c_out, edgecolor=c_out_border, linewidth=2)
    ax.add_patch(out_box)
    ax.text(14.5, 3.8, "Multi-Horizon Out", ha='center', va='center', fontsize=11, fontweight='bold', color="#991B1B")
    ax.text(14.5, 3.1, "[Batch, H, D_out]\nw[0]·yk + w[1]·yg + w[2]·ya\nMulti-step projection:\nH ∈ {1, 5, 10, ..., 60}", 
            ha='center', va='center', fontsize=8.5, color="#7F1D1D")

    # Connect Fusion to Output
    ax.annotate("", xy=(13.4, 3.4), xytext=(12.45, 3.4), arrowprops=dict(arrowstyle="->", color="#0F172A", lw=2))

    # Save
    out_dir = r"D:\user\projects\du_doan_gia_dau\Hybridtrinet_Oil\docs\images"
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "hybridtrinet_architecture.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out_path}")

if __name__ == "__main__":
    create_hybrid_trinet_diagram()

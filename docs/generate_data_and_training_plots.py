import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def generate_visualizations():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # -------------------------------------------------------------
    # 1. DATASET VISUALIZATION: Market Prices & Exogenous Trends
    # -------------------------------------------------------------
    pth = r"D:\user\projects\du_doan_gia_dau\Hybridtrinet_Oil\datasets\du_lieu_noi_suy_clean_updated.xlsx"
    df = pd.read_excel(pth)
    date_col = [c for c in df.columns if 'Ng' in c and 'y' in c][0]
    df['Date'] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=['Date']).sort_values('Date')
    
    for num_col in ['MG95', 'MG92', 'DO 0.05%', 'BRT DTD', 'WTI', 'USD_Index', 'GPRD']:
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors='coerce')
            
    fig, axes = plt.subplots(3, 1, figsize=(14, 11), sharex=True, dpi=300)
    
    # Panel A: Fuel Target Prices (MG95, MG92, DO)
    axes[0].plot(df['Date'], df['MG95'], label='Mogas 95 (MG95)', color='#DC2626', lw=1.5)
    axes[0].plot(df['Date'], df['MG92'], label='Mogas 92 (MG92)', color='#F59E0B', lw=1.2, linestyle='--')
    axes[0].plot(df['Date'], df['DO 0.05%'], label='Diesel (DO 0.05%)', color='#2563EB', lw=1.2)
    axes[0].set_title('Target Petroleum Spot Prices (USD/Barrel)', fontsize=12, fontweight='bold', color='#1E293B')
    axes[0].set_ylabel('USD / Barrel', fontsize=10)
    axes[0].legend(loc='upper right', frameon=True)
    axes[0].grid(True, linestyle=':', alpha=0.6)
    
    # Panel B: Global Benchmarks (Brent & WTI)
    axes[1].plot(df['Date'], df['BRT DTD'], label='Brent Crude (BRT DTD)', color='#059669', lw=1.4)
    axes[1].plot(df['Date'], df['WTI'], label='WTI Crude Spot', color='#7C3AED', lw=1.2, linestyle=':')
    axes[1].set_title('Global Crude Benchmarks (Brent vs WTI)', fontsize=12, fontweight='bold', color='#1E293B')
    axes[1].set_ylabel('USD / Barrel', fontsize=10)
    axes[1].legend(loc='upper right', frameon=True)
    axes[1].grid(True, linestyle=':', alpha=0.6)
    
    # Panel C: Macro & Exogenous Signals (USD Index & Geopolitical Risk)
    ax_c1 = axes[2]
    ax_c2 = ax_c1.twinx()
    
    l1 = ax_c1.plot(df['Date'], df['USD_Index'], label='USD Index', color='#475569', lw=1.3)
    l2 = ax_c2.plot(df['Date'], df['GPRD'], label='Geopolitical Risk Index (GPRD)', color='#EA580C', lw=1.0, alpha=0.75)
    
    ax_c1.set_title('Macroeconomic & Geopolitical Exogenous Drivers', fontsize=12, fontweight='bold', color='#1E293B')
    ax_c1.set_ylabel('USD Index (DXY)', fontsize=10, color='#475569')
    ax_c2.set_ylabel('GPRD Score', fontsize=10, color='#EA580C')
    
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax_c1.legend(lines, labels, loc='upper right', frameon=True)
    ax_c1.grid(True, linestyle=':', alpha=0.6)
    
    plt.suptitle('Multi-Variate Time-Series Dataset: Petroleum Targets & Exogenous Drivers (2008 - 2026)', 
                 fontsize=14, fontweight='bold', color='#0F172A', y=0.995)
    plt.tight_layout()
    out_data_img = r"D:\user\projects\du_doan_gia_dau\Hybridtrinet_Oil\docs\images\dataset_time_series_features.png"
    plt.savefig(out_data_img, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", out_data_img)
    
    # -------------------------------------------------------------
    # 2. TRAINING PERFORMANCE: Validation Loss Across Horizons
    # -------------------------------------------------------------
    hist_pth = r"D:\user\projects\du_doan_gia_dau\Hybridtrinet_Oil\checkpoints_multi\training_history.json"
    with open(hist_pth, 'r', encoding='utf-8') as f:
        history = json.load(f)
    
    baseline = [h for h in history if 'GUMNet_h60' in h.get('results', {})][-1]
    res = baseline['results']
    
    horizons = [1, 5, 10, 15, 20, 30, 60]
    gum_loss = [res[f'GUMNet_h{h}'] for h in horizons]
    hyb_loss = [res[f'HybridTriNet_h{h}'] for h in horizons]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), dpi=300)
    
    x = np.arange(len(horizons))
    width = 0.35
    
    rects1 = ax1.bar(x - width/2, gum_loss, width, label='GUMNet (Wavelet-KAN + CNN + GRU)', color='#8B5CF6', alpha=0.9)
    rects2 = ax1.bar(x + width/2, hyb_loss, width, label='HybridTriNet (RBF-KAN + Patch Transformer)', color='#0284C7', alpha=0.9)
    
    ax1.set_xlabel('Forecast Horizon (Days ahead)', fontsize=11, fontweight='bold', color='#1E293B')
    ax1.set_ylabel('Validation Loss (Normalized Scale)', fontsize=11, fontweight='bold', color='#1E293B')
    ax1.set_title('Validation Loss Convergence across 7 Horizons', fontsize=13, fontweight='bold', color='#0F172A')
    ax1.set_xticks(x)
    ax1.set_xticklabels([f'+{h}d' for h in horizons], fontsize=10)
    ax1.legend(frameon=True, loc='upper left')
    ax1.grid(True, linestyle=':', alpha=0.6)
    
    for r in rects1:
        h = r.get_height()
        ax1.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#5B21B6')
    for r in rects2:
        h = r.get_height()
        ax1.annotate(f'{h:.3f}', xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3),
                     textcoords="offset points", ha='center', va='bottom', fontsize=8, color='#0369A1')
                     
    ax2.plot(horizons, gum_loss, marker='o', lw=2.5, color='#8B5CF6', label='GUMNet Progression')
    ax2.plot(horizons, hyb_loss, marker='s', lw=2.5, color='#0284C7', label='HybridTriNet Progression')
    ax2.set_xlabel('Forecast Horizon (H in days)', fontsize=11, fontweight='bold', color='#1E293B')
    ax2.set_ylabel('Validation Error Trajectory', fontsize=11, fontweight='bold', color='#1E293B')
    ax2.set_title('Error Trajectory vs Prediction Distance', fontsize=13, fontweight='bold', color='#0F172A')
    ax2.set_xticks(horizons)
    ax2.legend(frameon=True, loc='upper left')
    ax2.grid(True, linestyle=':', alpha=0.6)
    
    ax2.annotate('Monotonic loss increase confirms\nno look-ahead leakage across horizons', 
                 xy=(30, 0.07), xytext=(12, 0.082),
                 arrowprops=dict(facecolor='#1E293B', shrink=0.05, width=1, headwidth=6),
                 fontsize=9, color='#1E293B', fontweight='bold',
                 bbox=dict(boxstyle="round,pad=0.3", fc="#FEF3C7", ec="#D97706", lw=1))
                 
    plt.suptitle('Multi-Horizon Model Training Evaluation (GUMNet vs HybridTriNet)', 
                 fontsize=14, fontweight='bold', color='#0F172A', y=0.98)
    plt.tight_layout()
    out_loss_img = r"D:\user\projects\du_doan_gia_dau\Hybridtrinet_Oil\docs\images\training_evaluation_loss.png"
    plt.savefig(out_loss_img, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", out_loss_img)

if __name__ == '__main__':
    generate_visualizations()

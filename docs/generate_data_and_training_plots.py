import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path

def generate_visualizations():
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    
    # -------------------------------------------------------------
    # 1. DATASET VISUALIZATION: Market Prices & Exogenous Trends
    # -------------------------------------------------------------
    root = Path(__file__).resolve().parents[1]
    pth = root / "datasets" / "du_lieu_noi_suy_clean_updated.xlsx"
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
    out_data_img = root / "docs" / "images" / "dataset_time_series_features.png"
    plt.savefig(out_data_img, dpi=300, bbox_inches='tight')
    plt.close()
    print("Saved:", out_data_img)
    
    # -------------------------------------------------------------
    # 2. CLEAN HOLDOUT PERFORMANCE: Same USD-scale metrics for both models
    # -------------------------------------------------------------
    metrics_path = root / "benchmark" / "clean_run" / "holdout_metrics.csv"
    metrics = pd.read_csv(metrics_path)
    horizons = [1, 5, 10, 15, 20, 30, 60]
    metric_specs = [
        ("mae", "MAE (USD/barrel)"),
        ("rmse", "RMSE (USD/barrel)"),
        ("mape", "MAPE (%)"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5), dpi=300)
    colors = {"GUMNet": "#8B5CF6", "HybridTriNet": "#0284C7"}
    x = np.arange(len(horizons))
    width = 0.36
    for ax, (metric_name, ylabel) in zip(axes, metric_specs):
        for offset, model_name in [(-width / 2, "GUMNet"), (width / 2, "HybridTriNet")]:
            subset = metrics[metrics["model"] == model_name].set_index("horizon")
            values = [float(subset.loc[h, metric_name]) for h in horizons]
            bars = ax.bar(x + offset, values, width, label=model_name, color=colors[model_name])
            ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
        ax.set_title(ylabel)
        ax.set_xlabel("Forecast horizon (days)")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x, [str(h) for h in horizons])
        ax.grid(axis="y", linestyle=":", alpha=0.55)
        ax.legend()

    fig.suptitle("Rolling-Origin Holdout Performance (70/15/15 split)", fontsize=14, fontweight="bold")
    fig.tight_layout()
    out_metrics_img = root / "docs" / "images" / "holdout_evaluation_metrics.png"
    plt.savefig(out_metrics_img, dpi=300, bbox_inches="tight")
    plt.close()
    print("Saved:", out_metrics_img)

if __name__ == '__main__':
    generate_visualizations()

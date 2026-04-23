import torch
import numpy as np
import matplotlib.pyplot as plt
import os
import seaborn as sns

def quantile_pinball_loss(preds, target, quantiles=[0.1, 0.5, 0.9]):
    """Tính toán Pinball Loss cho dự báo khoảng tin cậy"""
    target_expanded = target.unsqueeze(-1)
    losses = []
    for i, q in enumerate(quantiles):
        err = target_expanded[..., 0] - preds[..., i]
        loss = torch.max(q * err, (q - 1) * err)
        losses.append(loss)
    return torch.stack(losses, dim=-1).mean()

from sklearn.metrics import r2_score

def calculate_metrics(y_true, y_pred):
    """Tính toán các chỉ số đánh giá độ chính xác bao gồm R-squared"""
    epsilon = 1e-8
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(y_true - y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + epsilon))) * 100
    
    # Tính R-squared (Hệ số xác định)
    r2 = r2_score(y_true, y_pred)
    
    return {'MSE': mse, 'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}

def plot_q1_results(y_true, y_pred_10, y_pred_50, y_pred_90, weights_history, save_dir, target_names=['MG95', 'MG92', 'DO 0.001%', 'DO 0.05%']):
    """
    Xuất tập hợp các biểu đồ đánh giá chuyên sâu cho bài báo khoa học
    """
    os.makedirs(save_dir, exist_ok=True)
    y_true = np.array(y_true)
    y_pred_10 = np.array(y_pred_10)
    y_pred_50 = np.array(y_pred_50)
    y_pred_90 = np.array(y_pred_90)
    days = np.arange(len(y_true))

    # ---------------------------------------------------------
    # 1. BIỂU ĐỒ SO SÁNH 4 MẶT HÀNG (GRID 2x2)
    # ---------------------------------------------------------
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    axes = axes.flatten()
    for i in range(min(4, len(target_names))):
        axes[i].plot(days, y_true[:, i], color='black', label='Actual Price', linewidth=1.5)
        axes[i].plot(days, y_pred_50[:, i], color='red', label='GUM-Net Prediction', linestyle='--', linewidth=1.5)
        axes[i].fill_between(days, y_pred_10[:, i], y_pred_90[:, i], color='gray', alpha=0.2, label='90% Confidence Interval')
        axes[i].set_title(f'Petroleum Type: {target_names[i]}', fontsize=14, fontweight='bold')
        axes[i].set_xlabel('Days (Test Set)')
        axes[i].set_ylabel('Price (USD)')
        axes[i].legend(loc='best', fontsize=9)
        axes[i].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, '1_forecast_grid_4_targets.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 2. PHÂN PHỐI SAI SỐ (ERROR DISTRIBUTION)
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    for i in range(min(4, len(target_names))):
        error = y_true[:, i] - y_pred_50[:, i]
        sns.kdeplot(error, label=target_names[i], fill=True, alpha=0.3)
    
    plt.axvline(x=0, color='red', linestyle='--', alpha=0.6)
    plt.title('Prediction Error Distribution (Residual Analysis)', fontsize=14, fontweight='bold')
    plt.xlabel('Error Value (USD)')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.savefig(os.path.join(save_dir, '2_error_distribution.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 3. GATING DYNAMICS (TRỌNG SỐ CHUYÊN GIA)
    # ---------------------------------------------------------
    weights_np = np.array(weights_history)
    plt.figure(figsize=(12, 5))
    plt.stackplot(days, weights_np[:, 0], weights_np[:, 1], weights_np[:, 2], 
                  labels=['CNN (Local Fluctuations)', 'GRU (Long-term Trends)', 'Wav-KAN (Abrupt Shocks)'],
                  colors=['#4E79A7','#F28E2B','#59A14F'], alpha=0.85)
    plt.title('Gating Mechanism: Dynamic Expert Allocation over Time', fontsize=14, fontweight='bold')
    plt.xlabel('Days (Test Set)')
    plt.ylabel('Weight Ratio')
    plt.legend(loc='lower left', frameon=True, facecolor='white')
    plt.margins(0,0)
    plt.savefig(os.path.join(save_dir, '3_gating_weights_dynamics.png'), dpi=300)
    plt.close()

    # ---------------------------------------------------------
    # 4. SCATTER PLOT (CORRELATION ANALYSIS)
    # ---------------------------------------------------------
    plt.figure(figsize=(7, 7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    for i in range(min(4, len(target_names))):
        plt.scatter(y_true[:, i], y_pred_50[:, i], alpha=0.5, s=20, color=colors[i], label=target_names[i])
    
    # Đường 45 độ hoàn hảo
    min_val = min(np.min(y_true), np.min(y_pred_50))
    max_val = max(np.max(y_true), np.max(y_pred_50))
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', lw=2, label='Ideal Prediction')
    
    plt.title('Observed vs. Predicted Correlation', fontsize=14, fontweight='bold')
    plt.xlabel('Actual Values (USD)')
    plt.ylabel('Predicted Values (USD)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(os.path.join(save_dir, '4_scatter_correlation.png'), dpi=300)
    plt.close()

    print(f"✅ Đã xuất 4 biểu đồ chuyên sâu vào thư mục: {save_dir}")
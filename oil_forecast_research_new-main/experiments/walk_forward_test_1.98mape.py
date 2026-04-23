import sys, os, datetime
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import copy
import random
from tqdm import tqdm

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from src.model.dataset import DataProcessor, PetroleumDataset
from src.model.model import GUMNet
from src.model.utils import quantile_pinball_loss, calculate_metrics, plot_q1_results

# =====================================================================
# HÀM KHÓA SỰ NGẪU NHIÊN (REPRODUCIBILITY)
# =====================================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) 
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class DualLogger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding="utf-8")
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
        self.log.flush()
    def flush(self):
        self.terminal.flush()
        self.log.flush()

def run_experiment():
    set_seed(42)
    
    log_dir = os.path.join(project_root, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"run_exo_targets4_FINAL_{timestamp}.log")
    
    logger = DualLogger(log_file)
    sys.stdout = logger

    print(f"📄 Log đang lưu tại: {log_file}")
    print("🚀 CHIẾN DỊCH: DỰ BÁO ĐA BIẾN - CHỐT CẤU HÌNH HOÀNG KIM (CHỈ DÙNG GPR + USD)")
    
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"💻 Thiết bị tính toán: {DEVICE}")

    SEQ_LEN = 30
    HORIZON = 5
    TEST_DAYS = 100
    PATIENCE = 7
    MIN_EPOCHS = 15
    
    # 🌟 Nạp file Ver 1 (Chứa đủ GPR và USD Index chuẩn xác)
    data_path = os.path.join(project_root, 'data', 'processed', 'clean_data_exo_ver1.csv')
    df = pd.read_csv(data_path)
    
    target_cols = ['MG95', 'MG92', 'DO 0.001%', 'DO 0.05%']
    
    # 🌟 ĐÂY LÀ CHÌA KHÓA: Loại bỏ toàn bộ nhiễu FRED, chốt 13 biến tối ưu nhất!
    feature_cols = [
        'MG97', 'MG95', 'MG92', 'NAPHTHA', 'KERO ', 'DO 0.001%', 'DO 0.05%', 'FO 180', 
        'BRT DTD', 'BRT KH', 'WTI',  # Dữ liệu nội tại xuất sắc
        'USD_Index', 'GPR'           # 2 Biến vĩ mô hoàn hảo
    ]
    
    processor = DataProcessor(seq_len=SEQ_LEN, horizon=HORIZON)
    iterations = TEST_DAYS // HORIZON
    
    list_mape, list_mse, list_r2, list_rmse, list_mae = [], [], [], [], []
    all_true, all_pred_10, all_pred_50, all_pred_90, all_weights = [], [], [], [], []
    
    print("-" * 130)
    print(f"{'VÒNG':<6} | {'MAPE(%)':<8} | {'MSE':<8} | {'R2':<8} | {'RMSE':<8} | {'GATING (CNN/GRU/KAN)':<20} | {'EPOCH'}")
    print("-" * 130)
    
    for i in range(iterations):
        # Khởi tạo mô hình bên trong để True Walk-Forward
        model = GUMNet(seq_len=SEQ_LEN, input_dim=len(feature_cols), output_dim=len(target_cols), horizon=HORIZON).to(DEVICE)
        
        current_train_end = len(df) - TEST_DAYS + (i * HORIZON)
        
        train_df = df.iloc[:current_train_end]
        test_df = df.iloc[current_train_end - SEQ_LEN : current_train_end + HORIZON]
        
        X_all_train, y_all_train = processor.prepare_data(train_df, target_cols, feature_cols, is_train=True)
        X_test, _ = processor.prepare_data(test_df, target_cols, feature_cols, is_train=False)
        X_test_tensor = torch.tensor(X_test[-1:], dtype=torch.float32).to(DEVICE)
        
        split_idx = int(len(X_all_train) * 0.85)
        X_tr, y_tr = X_all_train[:split_idx], y_all_train[:split_idx]
        X_val, y_val = X_all_train[split_idx:], y_all_train[split_idx:]
        
        train_loader = DataLoader(PetroleumDataset(X_tr, y_tr), batch_size=64, shuffle=True)
        val_loader = DataLoader(PetroleumDataset(X_val, y_val), batch_size=64, shuffle=False)
        
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_model_wts = copy.deepcopy(model.state_dict())
        best_epoch = 0
        epoch = 0 
        
        while True:
            model.train()
            for batch_X, batch_y in train_loader:
                batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                optimizer.zero_grad()
                preds, _ = model(batch_X)
                loss = quantile_pinball_loss(preds, batch_y)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch_X, batch_y in val_loader:
                    batch_X, batch_y = batch_X.to(DEVICE), batch_y.to(DEVICE)
                    preds, _ = model(batch_X)
                    val_loss += quantile_pinball_loss(preds, batch_y).item()
            val_loss /= len(val_loader)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0 
                best_model_wts = copy.deepcopy(model.state_dict()) 
                best_epoch = epoch
            else:
                patience_counter += 1
                if epoch >= MIN_EPOCHS and patience_counter >= PATIENCE:
                    break
            epoch += 1 
                    
        model.load_state_dict(best_model_wts)
        model.eval()
        with torch.no_grad():
            pred_out, weights = model(X_test_tensor) 
            
            p_10_2d = processor.inverse_transform_targets(pred_out[0, :, :, 0].cpu().numpy())
            p_50_2d = processor.inverse_transform_targets(pred_out[0, :, :, 1].cpu().numpy())
            p_90_2d = processor.inverse_transform_targets(pred_out[0, :, :, 2].cpu().numpy())
            true_2d = test_df[target_cols].iloc[-HORIZON:].values
            
            m = calculate_metrics(true_2d.flatten(), p_50_2d.flatten())
            list_mape.append(m['MAPE'])
            list_mse.append(m['MSE'])
            list_r2.append(m['R2'])
            list_rmse.append(m['RMSE'])
            list_mae.append(m['MAE'])
            
            all_true.extend(true_2d)
            all_pred_10.extend(p_10_2d); all_pred_50.extend(p_50_2d); all_pred_90.extend(p_90_2d)
            for _ in range(HORIZON): all_weights.append(weights[0].cpu().numpy())
            
            w = weights[0].cpu().numpy()
            print(f"{i+1:02d}/{iterations:02d} | {m['MAPE']:<8.2f} | {m['MSE']:<8.2f} | {m['R2']:<8.3f} | {m['RMSE']:<8.2f} | {w[0]:.2f}/{w[1]:.2f}/{w[2]:.2f} | {best_epoch:2d}")

    # =================================================================================
    # TỔNG KẾT & XUẤT BÁO CÁO CHI TIẾT
    # =================================================================================
    print("-" * 130)
    print(f"🏆 KẾT QUẢ TỔNG HỢP (TRUNG BÌNH 4 MẶT HÀNG THÀNH PHẨM):")
    print(f"   - MAPE : {np.mean(list_mape):.3f}% | R-squared (R2): {np.mean(list_r2):.4f}")
    print(f"   - MSE  : {np.mean(list_mse):.4f} | RMSE : {np.mean(list_rmse):.3f} | MAE : {np.mean(list_mae):.3f}")
    
    all_true_np = np.array(all_true)
    all_pred_50_np = np.array(all_pred_50)
    summary_data = []
    
    for idx, name in enumerate(target_cols):
        y_t = all_true_np[:, idx]
        y_p = all_pred_50_np[:, idx]
        m = calculate_metrics(y_t, y_p)
        summary_data.append({
            'Mặt hàng': name,
            'MAPE (%)': f"{m['MAPE']:.3f}",
            'R2': f"{m['R2']:.4f}",
            'MSE': f"{m['MSE']:.4f}",
            'MAE': f"{m['MAE']:.3f}",
            'RMSE': f"{m['RMSE']:.3f}"
        })
    
    df_summary = pd.DataFrame(summary_data)
    print("\n📊 BẢNG CHI TIẾT TỪNG MẶT HÀNG (DÙNG CHO BÀI BÁO):")
    print(df_summary.to_string(index=False))
    
    # 🌟 Thư mục FINAL để báo cáo thầy
    save_dir = os.path.join(project_root, 'results/ver_FINAL_golden')
    os.makedirs(save_dir, exist_ok=True)
    df_summary.to_csv(os.path.join(save_dir, 'detailed_metrics_exo.csv'), index=False, encoding='utf-8-sig')
    print("-" * 130)
    print(f"✅ Đã lưu bảng phân tích chi tiết vào: {os.path.join(save_dir, 'detailed_metrics_exo.csv')}")

    print("\n🎨 Đang xuất biểu đồ phân tích đa biến (Grid 4 targets) vào /results ...")
    plot_q1_results(all_true, all_pred_10, all_pred_50, all_pred_90, all_weights, save_dir, target_names=target_cols)
    
    model_save_path = os.path.join(save_dir, 'gum_net_final_model.pth')
    torch.save(model.state_dict(), model_save_path)
    print(f"💾 Đã lưu trọng số mô hình tại: {model_save_path}")
    print("✅ Hoàn tất toàn bộ chiến dịch huấn luyện và đánh giá!")
    
    sys.stdout = logger.terminal

if __name__ == "__main__":
    run_experiment()
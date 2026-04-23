# 🛢️ Oil Price Forecast – GUMNet

Dự án ứng dụng mô hình Deep Learning tiên tiến (GUMNet: kết hợp CNN, GRU, WaveletKAN và Cơ chế Attention) để dự báo giá các mặt hàng xăng dầu trong nước (MG95, MG92, DO 0.001%, DO 0.05%) dựa trên các biến số vĩ mô và giá dầu thế giới.

---

## ✨ Các chức năng chính của Ứng dụng (Streamlit)

1. **🔮 Dự báo tương lai (Future Forecast):** 
   - Dự báo giá xăng dầu cho `N` ngày làm việc tiếp theo.
   - Hiển thị dải tin cậy với các đường bách phân vị (Quantile bands: **p10**, **p50**, **p90**). Đường p50 đóng vai trò là dự báo chính (Median), còn dải p10-p90 cho biết mức độ rủi ro/biến động có thể xảy ra.

2. **📊 So sánh với lịch sử (In-sample Backtest):**
   - Trượt cửa sổ (Sliding window) qua dữ liệu lịch sử để dự đoán bước tiếp theo (h=+1) và so sánh trực tiếp với giá trị thực tế đã xảy ra.
   - Trực quan hóa sai số bằng biểu đồ phân phối sai số (Residuals Histogram).

3. **✅ Đối chiếu với giá thực tế (Actual vs Predicted):**
   - Cho phép upload dữ liệu giá thực tế mới nhất tương ứng với khoảng thời gian đã được dự báo.
   - Hệ thống tự động ghép nối (merge) theo ngày và tính toán trực tiếp độ chính xác của lần dự báo đó.

4. **🧩 Tự động bù đắp dữ liệu (Auto-Merge Exogenous Variables):**
   - Khi dự báo, nếu file upload của người dùng chỉ có giá xăng dầu trong nước (thiếu các cột vĩ mô như `USD_Index`, `WTI`, `GPR`...), hệ thống sẽ tự động tìm kiếm và bù đắp các dữ liệu này từ Dataset chuẩn của hệ thống để đảm bảo mô hình chạy chính xác.

5. **🎯 Đa mục tiêu (Multi-Target Selection):**
   - Cho phép người dùng chuyển đổi linh hoạt qua lại giữa các mặt hàng dầu (MG95, MG92, DO...) chỉ với một cú click chuột.

---

## 📐 Các công thức đo lường độ chính xác (Metrics)

Ứng dụng sử dụng 3 chỉ số phổ biến nhất để đo lường mức độ sai lệch giữa giá dự đoán và giá thực tế. 

Ký hiệu:
- $y_i$: Giá trị thực tế tại ngày $i$.
- $\hat{y}_i$: Giá trị dự đoán (p50) tại ngày $i$.
- $n$: Số lượng ngày dự đoán.

### 1. MAE (Mean Absolute Error - Sai số tuyệt đối trung bình)
Thể hiện mức sai lệch trung bình về mặt con số tuyệt đối (đơn vị: VNĐ/lít hoặc USD/thùng). MAE càng nhỏ càng tốt.
> **Công thức:** 
> $$MAE = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|$$

### 2. RMSE (Root Mean Squared Error - Căn bậc hai sai số toàn phương trung bình)
Tương tự MAE nhưng RMSE phạt rất nặng các dự đoán có sai số lớn. Nếu RMSE lớn hơn MAE nhiều, chứng tỏ mô hình có những ngày dự đoán bị lệch cực kỳ nghiêm trọng.
> **Công thức:**
> $$RMSE = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}$$

### 3. MAPE (Mean Absolute Percentage Error - Sai số phần trăm tuyệt đối trung bình)
Thể hiện mức sai lệch dưới dạng phần trăm (%). MAPE giúp dễ hình dung mức độ sai lệch so với giá trị cốt lõi. (Ví dụ: MAPE = 2% nghĩa là dự báo lệch trung bình 2% so với giá trị thực).
> **Công thức:**
> $$MAPE = \frac{100\%}{n} \sum_{i=1}^{n} \left| \frac{y_i - \hat{y}_i}{y_i} \right|$$

*(Lưu ý: MAPE không tính các trường hợp giá trị thực tế $y_i = 0$ để tránh lỗi chia cho 0).*

---

## 📉 Hàm mất mát Quantile Loss (Dùng trong huấn luyện)

Trong quá trình huấn luyện (train), để mô hình có thể dự báo được các dải tin cậy (p10, p90), GUMNet sử dụng hàm **Quantile Loss** (hay còn gọi là Pinball Loss) thay vì MSE thông thường.

Với một mức phân vị $q$ (ví dụ: $q = 0.9$ cho p90), và sai số $e = y_i - \hat{y}_i$:
> **Công thức:**
> $$L_q(y, \hat{y}) = \max(q \cdot e, (q - 1) \cdot e)$$

Hệ thống sẽ tính tổng lỗi này trên cả 3 phân vị ($q \in \{0.1, 0.5, 0.9\}$) để tối ưu hóa trọng số (Gate weights) cho 3 nhánh CNN, GRU và WaveletKAN.

---

## ⚙️ Hướng dẫn cài đặt và chạy cục bộ

```bash
# 1. Cài đặt các thư viện cần thiết
pip install -r requirements.txt

# 2. Huấn luyện mô hình (Nếu cần update mặt hàng mới)
# (Có thể cấu hình TARGET_COLS trong file train.py trước)
python train.py

# 3. Chạy giao diện Web App
streamlit run app.py
```

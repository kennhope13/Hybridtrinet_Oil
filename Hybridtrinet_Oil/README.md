# HybridTriNet Oil Forecast

Ứng dụng **Streamlit** dự báo giá xăng dầu theo chuỗi thời gian bằng mô hình **HybridTriNet (KAN + GRU + Attention)**.  
Hỗ trợ **cập nhật dữ liệu bằng upload**, xử lý thiếu dữ liệu (NaN), nội suy một số biến ngoại sinh và **dự báo N ngày làm việc tiếp theo**, kèm dashboard so sánh **Dự báo vs Thực tế** và bảng **metrics**.

🔗 **Live Demo:** https://oil-price-prediction-hybridtrinet.streamlit.app/

---

## Highlights

- Dự báo đa biến cho các target: **MG95, MG92, DO 0.001%, DO 0.05%**
- Upload file `price_petroleum` để **gộp/cập nhật** dữ liệu
- Tiền xử lý: parse ngày, xử lý NaN, nội suy biến ngoại sinh (tuỳ cấu hình)
- Dự báo theo **N ngày làm việc tiếp theo** (business days)
- Lưu lịch sử dự báo (`forecast_history`) để so sánh theo nhiều kịch bản **H**
- Dashboard:
  - Candlestick dữ liệu nền
  - Actual vs Forecast theo nhiều horizon
  - Metrics: MAE / MAPE / MSE / RMSE / R²
  - Bảng đối chiếu chi tiết theo ngày

---

## Screenshots

<p align="center">
  <img src="images/BD_DATASET.png" width="95%" alt="Biểu đồ nền dữ liệu" />
</p>

<p align="center">
  <img src="images/DUDOANVSTHUCTE.png" width="95%" alt="Thực tế vs Dự đoán" />
</p>

<p align="center">
  <img src="images/CHISO.png" width="95%" alt="Metrics tổng hợp" />
</p>

<p align="center">
  <img src="images/BIEUDOSOSANH.png" width="95%" alt="Bảng so sánh theo target" />
</p>

<p align="center">
  <img src="images/UPLOAD.png" width="95%" alt="Upload và thiết lập dự đoán" />
</p>

---

## Yêu cầu dữ liệu

### File upload `price_petroleum`
- Định dạng: `.csv`, `.xlsx`, `.xls`
- Bắt buộc có cột ngày đúng tên với ô **“Cột ngày”** trong app (mặc định: `Ngày`)
- Khuyến nghị: dữ liệu theo ngày (daily)

---

## Cài đặt

### Python
Khuyến nghị **Python 3.11.14 pip 25.2**

### Cài thư viện
Nếu repo có `requirements.txt`:
```bash
pip install -r requirements.txt
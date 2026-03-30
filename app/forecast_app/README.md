# 📁 forecast_app

Thư mục `forecast_app` chứa các module cốt lõi phục vụ ứng dụng dự báo giá xăng dầu.  
Các file trong thư mục này được tách riêng theo từng chức năng để thuận tiện cho việc phát triển, bảo trì và mở rộng hệ thống.

---

## ✨ Chức năng của từng file

| File | Chức năng |
|------|-----------|
| `app/app_forecast.py` | File chạy chính của ứng dụng Streamlit. Phụ trách dựng giao diện, nhận dữ liệu đầu vào từ người dùng, gọi pipeline xử lý và hiển thị kết quả dự báo. |
| `forecast_app/__init__.py` | File khởi tạo package `forecast_app`, cho phép các module trong thư mục import lẫn nhau theo cấu trúc package Python. |
| `forecast_app/config.py` | Chứa các cấu hình dùng chung cho toàn bộ ứng dụng như danh sách biến mục tiêu, số bước nhìn lại, số bước dự báo và các tham số mặc định khác. |
| `forecast_app/style.py` | Quản lý phần giao diện hiển thị, dùng để chèn CSS và tùy chỉnh phong cách trình bày của ứng dụng Streamlit. |
| `forecast_app/ui.py` | Chứa các thành phần giao diện dùng lại như tiêu đề trang, tiêu đề mục, đường phân cách và các khối hiển thị chuẩn hóa. |
| `forecast_app/data_helpers.py` | Hỗ trợ đọc và tiền xử lý dữ liệu đầu vào, bao gồm đọc file actual và file upload, chuẩn hóa cột ngày, xử lý giá trị thiếu, nội suy và gộp dữ liệu. |
| `forecast_app/plots.py` | Chứa các hàm vẽ biểu đồ, phục vụ trực quan hóa dữ liệu thực tế và dữ liệu dự báo để hỗ trợ so sánh và phân tích kết quả. |
| `forecast_app/train_focus5.py` | Phụ trách huấn luyện mô hình dự báo, với định hướng tối ưu tốt hơn cho các bước dự báo ngắn hạn, đặc biệt là 5 ngày đầu. |
| `forecast_app/autoregressive.py` | Triển khai cơ chế dự báo tự hồi quy nhiều bước, sử dụng kết quả dự báo trước đó làm đầu vào cho các bước tiếp theo. |
| `forecast_app/calibration.py` | Thực hiện hiệu chỉnh kết quả dự báo nhằm giảm sai lệch giữa forecast và actual dựa trên dữ liệu lịch sử. |
| `forecast_app/metrics.py` | Tính toán các chỉ số đánh giá mô hình như MAE, MAPE, MSE, RMSE, R² và các thống kê liên quan khác. |
| `forecast_app/history_eval.py` | Đánh giá lịch sử dự báo bằng cách đọc các file forecast trước đó, so sánh với actual hiện tại và tính metrics trên vùng dữ liệu trùng nhau. |
| `forecast_app/core.py` | Điều phối pipeline chính của hệ thống, kết nối các bước xử lý dữ liệu, huấn luyện, dự báo, hiệu chỉnh và đánh giá kết quả. |

---

## 🧩 Tóm tắt

Nhìn tổng thể, thư mục `forecast_app` được chia thành 4 nhóm chức năng chính:

- **Giao diện:** `app_forecast.py`, `style.py`, `ui.py`
- **Cấu hình và dữ liệu:** `config.py`, `data_helpers.py`
- **Huấn luyện và dự báo:** `train_focus5.py`, `autoregressive.py`, `calibration.py`, `core.py`
- **Đánh giá và trực quan hóa:** `metrics.py`, `history_eval.py`, `plots.py`

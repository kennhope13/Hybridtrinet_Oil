# HybridTriNet Oil Forecast – Streamlit (1-file app)

Ứng dụng Streamlit dự báo giá xăng dầu theo chuỗi thời gian bằng mô hình **HybridTriNet**. App hỗ trợ **đọc dữ liệu gốc**, **upload file price_petroleum để gộp/cập nhật**, xử lý NaN, nội suy một số biến ngoại sinh và **dự báo N ngày làm việc tiếp theo**, kèm biểu đồ so sánh **Dự báo vs Thực tế**.

---

## Yêu cầu dữ liệu


### File upload `price_petroleum`
- Định dạng: `.csv`, `.xlsx`, `.xls`
- Bắt buộc có cột ngày đúng tên với ô **“Cột ngày”** trong app (mặc định: `Ngày`)

---

## Cài đặt

### Python
Khuyến nghị Python 3.10+.

### Cài thư viện
Nếu bạn chưa có `requirements.txt`, tối thiểu:
```bash
pip install streamlit torch pandas numpy openpyxl xlrd altair

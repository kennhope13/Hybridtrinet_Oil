
---

  **Mở rộng chân trời dự báo:**
    *   Tích hợp đầy đủ các mốc dự báo: **1 ngày, 5 ngày, 10 ngày, 30 ngày, 60 ngày và 100 ngày**.
  **Tối ưu hóa hiển thị:**
    *   Lọc dữ liệu: Chỉ hiển thị dữ liệu tập trung vào **Các file gần nhất** (từ tháng 09/2025) để loại bỏ nhiễu lịch sử.
    *   Ẩn cột "Ngày thứ" để bảng biểu gọn gàng, tập trung vào con số giá.
  **Tính năng Dự báo Trực tiếp (Live Forecast):**
    *   Hệ thống tự động lấy dữ liệu mới nhất từ file bạn vừa upload để phóng tầm mắt ra **100 ngày tới**.
    *   Vẽ biểu đồ **lộ trình đầy đủ 100 điểm** thay vì chỉ hiện các mốc rời rạc.

---

## ✨ Các tính năng chính

*   **🏆 Tổng kết sai số (MAE/MAPE):** Tự động tính trung bình sai số của tất cả các file đã upload để đánh giá mô hình nào "thông minh" hơn.
*   **🔮 Dự báo đa chân trời:** Cung cấp bảng giá cụ thể cho từng mốc (1d -> 100d) ngay sau khi upload file.
*   **📈 Biểu đồ so sánh:**
    *   So sánh Dự báo vs Thực tế trong quá khứ.
    *   So sánh lộ trình của 6 mô hình khác nhau (1d, 5d... 100d) trên cùng một biểu đồ tương lai.
*   **📂 Quản lý file tự động:** Mọi file bạn upload đều được lưu vào thư mục `datasets` và tự động nạp vào hệ thống cho những lần sử dụng sau.

---

## 📖 Hướng dẫn sử dụng

### Bước 1: Chuẩn bị dữ liệu
File của bạn (Excel hoặc CSV) cần có ít nhất các cột:
*   **Ngày:** Định dạng ngày tháng (ví dụ: 12/04/2026).
*   **Giá các mặt hàng:** MG95, MG92, DO 0.001%, DO 0.05%...

### Bước 2: Upload và Xem kết quả
1.  Truy cập tab **"⬆️ Upload & Dự báo"**.
2.  Nhấn **"Browse files"** và chọn file dữ liệu mới nhất.
3.  **Xem ngay:**
    *   Bảng dự báo giá cho ngày mai và 100 ngày tới.
    *   Biểu đồ lộ trình biến động tương lai.

### Bước 3: Đánh giá mô hình
1.  Chuyển sang tab **"🏆 Tổng kết"** để xem sai số trung bình.
2.  Chuyển sang tab **"📈 Biểu đồ"** để nhìn lại những lần AI đoán đúng hoặc sai so với giá thực tế.

### Bước 4: Tùy chỉnh (Sidebar)
*   **Chọn mô hình:** Bạn có thể chọn chỉ xem GUMNet, chỉ xem HybridTriNet hoặc xem cả hai để đối chiếu.
*   **Xóa Cache:** Nếu bạn muốn hệ thống tính toán lại từ đầu toàn bộ 15 file, hãy nhấn nút **"🗑️ Xóa Cache Simulation"**.

---

## 🛠️ Lưu ý kỹ thuật
*   Ứng dụng chạy trên nền tảng **Streamlit**.
*   Các mô hình AI được nạp từ thư mục `checkpoints_multi`.
*   Dữ liệu gốc được lưu tại `data/processed/clean_data_exo_ver1.csv`.

---
*Chúc bạn có những phân tích và dự báo chính xác!*

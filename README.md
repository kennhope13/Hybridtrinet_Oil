# 🛢️ Hệ Thống Dự Báo Giá Dầu Khí Thông Minh (AI Oil Forecast)

Ứng dụng này sử dụng các mô hình Trí Tuệ Nhân Tạo (Deep Learning) tiên tiến như **GUMNet** và **HybridTriNet** để học quy luật thị trường và đưa ra các dự báo giá dầu mỏ cho nhiều mốc thời gian trong tương lai.

---

## 📅 1. Cách Tính Ngày Dự Đoán (Lịch Trình Tương Lai)

Hệ thống được thiết kế để sát với thực tế giao dịch của thị trường tài chính và dầu mỏ:
*   **Bỏ qua cuối tuần:** Khi dự báo tương lai (1 ngày, 5 ngày, 100 ngày...), hệ thống tự động bỏ qua **Thứ 7 và Chủ Nhật**.
*   **Ví dụ:** Nếu ngày gốc (ngày upload cuối cùng) là Thứ Tư (22/04), thì:
    *   Mốc **+1 ngày**: sẽ rơi vào Thứ Năm (23/04).
    *   Mốc **+5 ngày**: sẽ rơi vào Thứ Tư tuần sau (29/04) vì đã tự động bỏ qua ngày 25 và 26 (T7, CN).
*   **Điểm Neo (Từ Mốc):** Khi bạn upload một file, hệ thống sẽ BẮT BUỘC lấy ngày lớn nhất trong file đó làm "Hiện tại" và dùng nó để phóng chiếu ra tương lai.

---

## 🧮 2. Cách Tính Các Chỉ Số Đánh Giá Mô Hình

Hệ thống đo lường độ thông minh của AI thông qua các chỉ số sai số chuẩn quốc tế:

*   **MAE (Mean Absolute Error - Sai số tuyệt đối trung bình):** Đo lường mức độ lệch giá trị thực tế (USD). Nếu MAE = 2.5 nghĩa là trung bình mô hình đoán chênh lệch 2.5 đô la so với thực tế.
    *   *Cách tính:* Tổng các độ lệch tuyệt đối chia cho số lượng dự đoán.
*   **MAPE (Mean Absolute Percentage Error - Sai số phần trăm trung bình):** Đo lường tỷ lệ phần trăm sai lệch. Đây là chỉ số quan trọng nhất để xem mô hình có an toàn không. (Cảnh báo đỏ nếu MAPE > 10%).
    *   *Cách tính:* Tính phần trăm sai số của từng ngày `(|Thực tế - Dự báo| / Thực tế) * 100`, rồi lấy trung bình.
*   **MSE & RMSE (Sai số bình phương):** Trừng phạt nặng các dự đoán sai lệch lớn. (Được tích hợp ngầm trong quá trình huấn luyện/finetune).

---

## ⚙️ 3. Quy Trình Hoạt Động (Workflow)

Khi bạn thao tác trên ứng dụng, AI hoạt động theo quy trình sau:
1.  **Nạp Dữ liệu Gốc:** Nạp file lịch sử dài hạn (clean_data_exo_ver1.csv) có chứa các biến ngoại sinh (Vàng, USD, Chỉ số địa chính trị).
2.  **Nhận Dữ liệu Upload:** Ghép file Excel/CSV bạn vừa tải lên vào dữ liệu gốc.
3.  **Lọc Mốc Tương Lai:** Cắt bỏ dữ liệu tương lai rác (nếu có) để ép hệ thống hiểu rằng ngày cuối cùng trong file Upload chính là "Hôm nay".
4.  **Chuẩn bị (Scaling):** Chuẩn hóa (Normalize) dữ liệu để tương thích với nếp nhăn của mạng Neural.
5.  **Dự Báo & Trực Quan Hóa:** Đẩy qua mô hình GUMNet/HybridTriNet để xuất ra giá trị tương lai và vẽ biểu đồ.
6.  **Tự Động Finetune (Nếu Bật):** Mở tiến trình chạy ngầm để AI cập nhật lại trọng số (Weights) dựa trên vùng giá mới (giúp tránh lỗi mô hình bị ngáo khi giá nhảy vọt).

---

## 🗂️ 4. Ý Nghĩa Của Các Tab (Thẻ)

Ứng dụng chia làm 5 khu vực chuyên biệt:

### ⬆️ Tab 1: Upload & Dự Báo (Live Forecast)
*   **Công dụng:** Trạm điều khiển chính. Bạn upload dữ liệu mới nhất vào đây để xem dự báo cho Tương lai (1-100 ngày tới).
*   **Tính năng đặc biệt:** Bảng tóm tắt đa mốc thời gian và Bảng chi tiết liệt kê lộ trình từng ngày. Có nút Tích tự động cập nhật độ thông minh cho AI ngay khi tải file lên.

### 🏆 Tab 2: Tổng kết (Backtest Metrics)
*   **Công dụng:** Sổ học bạ của AI. Dùng để xem AI đã dự đoán quá khứ đúng hay sai.
*   **Tính năng đặc biệt:** Bảng MAE, MAPE và Hệ thống Cảnh báo Thông minh (hiển thị Đỏ/Vàng/Xanh tùy vào mức độ sai số).

### 📋 Tab 3: Lịch sử upload
*   **Công dụng:** Sổ cái ghi chép. Xem lại chi tiết từng con số Dự báo vs Thực tế cho từng đợt dữ liệu bạn đã từng upload vào ứng dụng.

### 📈 Tab 4: Biểu đồ (Visualization)
*   **Công dụng:** Góc nhìn trực quan. Giúp so sánh xu hướng (Trend) giữa đường dự đoán của 2 mô hình (nét đứt) và đường giá thực tế (nét liền xanh).

### 🗃️ Tab 5: Bảng chi tiết
*   **Công dụng:** Dành cho dân data. Xuất toàn bộ DataFrame kết quả chạy mô phỏng để copy/paste hoặc phân tích chuyên sâu bên ngoài.

---
*Cẩm nang này giúp bạn nắm quyền kiểm soát hoàn toàn hệ thống AI Oil Forecast!*

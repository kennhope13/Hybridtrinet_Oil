# Hướng dẫn ngắn: Dữ liệu cần nhập & Ý nghĩa kết quả

> Tài liệu này dành cho người dùng không rành kỹ thuật, giải thích ngắn gọn cách chuẩn bị file dữ liệu và cách đọc hiểu kết quả dự báo/đánh giá của hệ thống.

## 1. File dữ liệu cần chuẩn bị

- **Định dạng chấp nhận:** Excel (`.xlsx`, `.xls`) hoặc CSV (`.csv`). Dung lượng tối đa **200 MB/file**.
- **Cột bắt buộc:**
  - 1 cột **Ngày** (tên cột có thể là "Ngày", "ngay", hoặc "date").
  - Ít nhất 1 trong các cột giá: `MG95`, `MG92`, `DO 0.001%`, `DO 0.05%`.
- **Mỗi dòng là 1 ngày.** Nếu file có 2 dòng trùng ngày, hệ thống tự lấy dòng **sau cùng** (dòng nằm dưới trong file) làm giá trị chính thức.
- **Ô để trống (thiếu giá)** ở giữa file sẽ được hệ thống tự động nội suy (ước lượng dựa trên các ngày lân cận) — không cần tự điền tay.
- **Không cần sắp xếp theo ngày trước khi tải lên** — hệ thống tự sắp xếp lại.

## 2. Sau khi tải file lên

1. Hệ thống hiện trước ngày bắt đầu/kết thúc của file để bạn kiểm tra đúng file chưa — **chưa lưu gì cả** ở bước này.
2. Bấm **"⚙️ Xử lý"** để chính thức lưu và tính dự báo mới.
3. Nếu dữ liệu có ngày **mới hơn** hệ thống đang có, sẽ có gợi ý sang trang Huấn luyện để cập nhật lại mô hình (không bắt buộc, có thể bỏ qua và chỉ xem dự báo).

## 3. Ý nghĩa các chỉ số ở trang "Đánh giá mô hình"

| Chỉ số | Ý nghĩa | Ngưỡng tham khảo |
|---|---|---|
| **MAPE (%)** | Phần trăm sai lệch trung bình giữa giá AI dự đoán và giá thị trường thực tế | Xanh (&lt;7%): rất tốt · Vàng (7–10%): chấp nhận được · Đỏ (&gt;10%): nên huấn luyện lại |
| **MAE (USD)** | Sai số tuyệt đối tính bằng đúng đơn vị tiền tệ/giá (USD/thùng) | Càng nhỏ càng chính xác |
| **Số điểm dữ liệu** | Số lượng lần đối chiếu dự báo cũ vs giá thực tế đã diễn ra | Càng nhiều, con số MAPE/MAE càng đáng tin cậy |

## 4. Khi nào nên huấn luyện lại (Finetune)?

- Định kỳ khoảng 1 tháng/lần, sau khi đã có đủ dữ liệu giá thực tế của tháng đó.
- Hoặc ngay khi trang "Đánh giá mô hình" báo MAPE vượt quá 10% (vùng đỏ).
- Huấn luyện lại **không** làm mất dữ liệu lịch sử đã học — chỉ học thêm ("Finetune") theo dữ liệu mới, trừ khi bạn chủ động chọn "Huấn luyện lại từ đầu".

## 5. Nếu có lỗi

- Hệ thống **luôn giữ nguyên mô hình cũ** cho tới khi mô hình mới được kiểm tra đầy đủ và chắc chắn tốt — huấn luyện lỗi giữa chừng **không làm hỏng** hay mất mô hình đang dùng.
- Mỗi lần huấn luyện, hệ thống tự sao lưu lại bản trước đó (giữ 5 bản gần nhất) — nếu bản mới không tốt, có thể khôi phục lại ở tab "Lịch sử & Dữ liệu đầu vào" trong trang Huấn luyện.
- Nếu gặp thông báo lỗi, có nút **"🔁 Thử lại"** — nếu vẫn lỗi sau khi thử lại, liên hệ người phụ trách kỹ thuật kèm ảnh chụp màn hình.

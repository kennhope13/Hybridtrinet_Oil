# Chức năng các file trong thư mục `forecast_app`

### `app.py`
File chạy chính của ứng dụng Streamlit.  
Dùng để tạo giao diện, nhận dữ liệu người dùng, gọi các hàm xử lý dữ liệu, huấn luyện, dự báo và hiển thị kết quả.

### `core.py`
File điều phối pipeline chính của hệ thống.  
Kết nối các bước từ xử lý dữ liệu, huấn luyện mô hình, dự báo, hiệu chỉnh đến đánh giá kết quả.

### `config.py`
File cấu hình chung của ứng dụng.  
Chứa các tham số mặc định như danh sách target dự báo, số bước nhìn lại, số bước dự báo và các thiết lập khác.

### `style.py`
File quản lý phần hiển thị giao diện.  
Dùng để chèn CSS và tùy chỉnh giao diện Streamlit cho đồng bộ và dễ nhìn hơn.

### `ui.py`
File chứa các thành phần giao diện dùng lại.  
Ví dụ như tiêu đề trang, tiêu đề mục, đường phân cách hoặc các khối hiển thị chuẩn.

### `data_helpers.py`
File hỗ trợ đọc và xử lý dữ liệu đầu vào.  
Dùng để đọc file actual/upload, chuẩn hóa cột ngày, xử lý thiếu dữ liệu, nội suy và gộp dữ liệu mới với dữ liệu cũ.

### `plots.py`
File phục vụ vẽ biểu đồ.  
Dùng để trực quan hóa dữ liệu thực tế và dữ báo, hỗ trợ so sánh xu hướng giữa actual và forecast.

### `train_focus5.py`
File huấn luyện mô hình dự báo.  
Tập trung tối ưu chất lượng dự báo, đặc biệt cho các ngày đầu của horizon.

### `autoregressive.py`
File thực hiện dự báo tự hồi quy nhiều bước.  
Dùng output của bước trước làm input cho bước sau để mở rộng dự báo cho horizon dài hơn.

### `calibration.py`
File hiệu chỉnh kết quả dự báo.  
Dùng để giảm độ lệch giữa forecast và actual bằng cách áp dụng các hệ số calibration học từ lịch sử.

### `metrics.py`
File tính các chỉ số đánh giá mô hình.  
Bao gồm MAE, MAPE, MSE, RMSE, R² và các chỉ số liên quan khác.

### `history_eval.py`
File đánh giá lịch sử dự báo.  
Dùng để đọc các file forecast cũ, so sánh với actual hiện tại và tính metrics trên vùng dữ liệu trùng nhau.

### `__init__.py`
File đánh dấu thư mục là một Python package.  
Giúp các module trong thư mục có thể import lẫn nhau theo cấu trúc package.
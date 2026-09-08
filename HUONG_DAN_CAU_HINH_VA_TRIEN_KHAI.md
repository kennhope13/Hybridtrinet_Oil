# TÀI LIỆU HƯỚNG DẪN CẤU HÌNH VÀ TRIỂN KHAI HỆ THỐNG DỰ BÁO GIÁ DẦU

* **Hệ thống:** Oil Forecast – Automated Evaluation Hub
* **Môi trường áp dụng:** Microsoft Windows Server 2016, 2019, 2022 hoặc Windows 10, 11
* **Đối tượng sử dụng:** Cán bộ quản trị CNTT và người dùng triển khai hệ thống

---

## MỤC LỤC
1. [Khuyến nghị cấu hình phần cứng](#1-khuyến-nghị-cấu-hình-phần-cứng)
2. [Cách 1: Triển khai từng bước cho máy mới](#2-cách-1-triển-khai-từng-bước-cho-máy-mới)
3. [Cách 2: Triển khai tự động bằng dòng lệnh PowerShell](#3-cách-2-triển-khai-tự-động-bằng-dòng-lệnh-powershell)
4. [Cách 3: Triển khai cho máy chủ nội bộ không có kết nối Internet](#4-cách-3-triển-khai-cho-máy-chủ-nội-bộ-không-có-kết-nối-internet)
5. [Mở cổng tường lửa để các máy trong phòng ban cùng truy cập](#5-mở-cổng-tường-lửa-để-các-máy-trong-phòng-ban-cùng-truy-cập)
6. [Cài đặt ứng dụng tự bật mỗi khi khởi động máy chủ](#6-cài-đặt-ứng-dụng-tự-bật-mỗi-khi-khởi-động-máy-chủ)
7. [Xử lý sự cố thường gặp](#7-xử-lý-sự-cố-thường-gặp)

---

## 1. KHUYẾN NGHỊ CẤU HÌNH PHẦN CỨNG

Hệ thống có thể chạy trên cả máy chủ ảo dùng CPU lẫn máy tính có card đồ họa rời:

| Thành phần | Máy chủ CPU dùng máy ảo hoặc VPS | Máy chủ có card đồ họa GPU |
| :--- | :--- | :--- |
| **Hệ điều hành** | Windows Server hoặc Windows 10, 11 bản 64-bit | Windows Server hoặc Windows 10, 11 bản 64-bit |
| **Bộ vi xử lý CPU** | 4 đến 6 nhân trở lên | 8 nhân trở lên |
| **Bộ nhớ RAM** | 16 GB | 16 GB |
| **Dung lượng ổ đĩa** | Trống từ 10 GB trở lên trên ổ SSD | Trống từ 10 GB trở lên trên ổ SSD |
| **Card đồ họa GPU** | Không bắt buộc, phần mềm tự chạy bằng CPU | Card NVIDIA GeForce hoặc RTX, bộ nhớ VRAM từ 4 GB đến 6 GB trở lên |
| **Khả năng dự báo** | Đáp ứng tốt nhu cầu khai thác hàng ngày | Tối ưu hóa hiệu năng tính toán |
| **Khả năng huấn luyện** | Phù hợp cho việc cập nhật định kỳ | Tận dụng năng lực xử lý song song |

---

## 2. CÁCH 1: TRIỂN KHAI TỪNG BƯỚC CHO MÁY MỚI

Thực hiện các bước sau để thiết lập máy chủ:

### Bước 2.1: Tải và cài đặt gói bổ trợ Microsoft Visual C++
1. Mở trình duyệt web Chrome hoặc Edge, truy cập đường dẫn:  
   `https://aka.ms/vs/17/release/vc_redist.x64.exe`
2. Mở tệp vừa tải về mang tên `vc_redist.x64.exe`.
3. Tích chọn đồng ý điều khoản, bấm nút **Install**, sau đó bấm **Close** khi hoàn tất.

### Bước 2.2: Tải và cài đặt Python 3.11
1. Truy cập đường dẫn tải Python chính thức:  
   `https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe`
2. Mở tệp cài đặt `python-3.11.9-amd64.exe`.
3. **Lưu ý quan trọng:** Tại màn hình đầu tiên, tích chọn ô **Add python.exe to PATH** ở góc dưới cùng.
4. Bấm nút **Install Now** và chờ phần mềm cài đặt hoàn tất, sau đó bấm **Close**.

### Bước 2.3: Cài đặt Git và đồng bộ mã nguồn bằng git clone
Để tải trọn vẹn 100% các tệp trọng số mô hình trong thư mục `checkpoints_multi` mà không bị thiếu tệp (hiện tượng tệp con trỏ 1 KB khi tải ZIP thủ công), hệ thống khuyến nghị sử dụng công cụ Git:

1. **Tải và cài đặt Git bằng chuột:**
   * Mở trình duyệt web truy cập đường dẫn tải Git chính thức:  
     `https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe`
   * Mở tệp vừa tải về mang tên `Git-2.47.1-64-bit.exe`, bấm nút **Next** theo mặc định cho đến khi bấm **Install** và hoàn tất.
2. **Dùng lệnh git clone kéo toàn bộ mã nguồn và tệp mô hình về máy:**
   * Bấm phím Windows, gõ **cmd** và mở cửa sổ Command Prompt.
   * Tạo và di chuyển vào thư mục làm việc, ví dụ trên ổ đĩa D:
     ```cmd
     mkdir D:\App_DuBaoGiaDau
     cd /d D:\App_DuBaoGiaDau
     git clone https://github.com/kennhope13/Hybridtrinet_Oil.git .
     ```
   * Dấu chấm ở cuối câu lệnh giúp đưa toàn bộ mã nguồn và các tệp mô hình trực tiếp vào thư mục vừa tạo. Công cụ Git sẽ tự động tải đầy đủ toàn bộ tệp trọng số mô hình `.pt` nguyên vẹn.
3. **Trường hợp sao chép thư mục được bàn giao:**
   * Nếu bạn đã nhận được trọn gói thư mục dự án qua mạng nội bộ hoặc ổ cứng di động, chỉ cần sao chép thư mục đó vào máy tính mà không cần tải lại từ GitHub.

### Bước 2.4: Khởi chạy ứng dụng
1. Mở thư mục ứng dụng vừa giải nén hoặc sao chép.
2. Tìm tệp tin mang tên:  
   👉 **CHAY_UNG_DUNG.bat**
3. Nhấp đúp chuột vào tệp này. Hệ thống sẽ tự động khởi tạo môi trường và mở trang web làm việc tại địa chỉ `http://localhost:8502`.

---

## 3. CÁCH 2: TRIỂN KHAI TỰ ĐỘNG BẰNG POWERSHELL

Mở PowerShell bằng quyền Quản trị viên (Run as administrator) và thực hiện:

### Bước 3.1: Cài đặt Visual C++ Runtime và Python tự động
```powershell
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12
Invoke-WebRequest -Uri "https://aka.ms/vs/17/release/vc_redist.x64.exe" -OutFile "$env:TEMP\vc_redist.exe"
Start-Process -FilePath "$env:TEMP\vc_redist.exe" -ArgumentList "/install", "/passive", "/norestart" -Wait
Remove-Item "$env:TEMP\vc_redist.exe"

Invoke-WebRequest -Uri "https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe" -OutFile "$env:TEMP\python_setup.exe"
Start-Process -FilePath "$env:TEMP\python_setup.exe" -ArgumentList "/quiet", "InstallAllUsers=1", "PrependPath=1", "Include_test=0" -Wait
Remove-Item "$env:TEMP\python_setup.exe"
```

### Bước 3.2: Tự động cài Git, kéo mã nguồn và khởi động
```powershell
# Cài đặt Git tự động
Invoke-WebRequest -Uri "https://github.com/git-for-windows/git/releases/download/v2.47.1.windows.1/Git-2.47.1-64-bit.exe" -OutFile "$env:TEMP\git_setup.exe"
Start-Process -FilePath "$env:TEMP\git_setup.exe" -ArgumentList "/VERYSILENT", "/NORESTART" -Wait
Remove-Item "$env:TEMP\git_setup.exe"

# Cập nhật đường dẫn biến môi trường
$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

# Tạo thư mục và đồng bộ toàn bộ mã nguồn từ GitHub
New-Item -ItemType Directory -Force -Path "D:\App_DuBaoGiaDau"
Set-Location "D:\App_DuBaoGiaDau"
git clone https://github.com/kennhope13/Hybridtrinet_Oil.git .
.\CHAY_UNG_DUNG.bat
```

---

## 4. CÁCH 3: TRIỂN KHAI CHO MÁY CHỦ NỘI BỘ KHÔNG CÓ KẾT NỐI INTERNET

Áp dụng cho các máy chủ bảo mật nghiêm ngặt, không được kết nối ra ngoài:

1. **Chuẩn bị trên máy có mạng:** Chạy tệp CHAY_UNG_DUNG.bat một lần để máy tải đủ thư mục venv và các tệp mô hình trong thư mục checkpoints_multi.
2. **Đóng gói:** Nén toàn bộ thư mục dự án cùng tệp cài đặt vc_redist.x64.exe thành tệp zip.
3. **Triển khai tại máy nội bộ:** Chuyển tệp zip sang máy chủ, giải nén và nhấp đúp vào vc_redist.x64.exe để cài gói bổ trợ nếu máy chưa có. Sau đó chỉ cần nhấp đúp vào CHAY_UNG_DUNG.bat để sử dụng mà không cần mạng.

---

## 5. MỞ CỔNG TƯỜNG LỬA ĐỂ CÁC MÁY TRONG PHÒNG BAN CÙNG TRUY CẬP

Sau khi cài đặt xong trên máy chủ, bạn có thể thực hiện bằng giao diện đồ họa chuột hoặc lệnh ngắn:

### Thao tác bằng chuột qua Windows Defender Firewall:
1. Nhấn phím Windows, gõ tìm kiếm **Windows Defender Firewall** và mở lên.
2. Nhìn vào cột bên trái, chọn **Advanced settings**.
3. Bấm chuột phải vào mục **Inbound Rules** ở góc trên bên trái, chọn **New Rule...**.
4. Chọn loại quy tắc là **Port**, bấm **Next**.
5. Nhập số cổng **8502** vào ô Specific local ports, bấm **Next**.
6. Chọn **Allow the connection**, bấm **Next**.
7. Đặt tên quy tắc là **Oil Forecast Web**, bấm **Finish**.

### Truy cập từ các máy tính khác:
Mở trình duyệt web trên máy cá nhân, gõ địa chỉ IP của máy chủ kèm cổng 8502, ví dụ: `http://192.168.1.100:8502`.

---

## 6. CÀI ĐẶT ỨNG DỤNG TỰ BẬT MỖI KHI KHỞI ĐỘNG MÁY CHỦ

Thao tác hoàn toàn bằng chuột qua công cụ Task Scheduler có sẵn của Windows:

1. Nhấn tổ hợp phím Windows + R, gõ `taskschd.msc` và bấm **OK**.
2. Tại cột bên phải, chọn mục **Create Basic Task...**.
3. Đặt tên tác vụ là **OilForecastHub**, bấm **Next**.
4. Chọn mục **When the computer starts** để tự chạy khi mở máy, bấm **Next**.
5. Chọn **Start a program**, bấm **Next**.
6. Tại ô Program/script, bấm nút **Browse...** và trỏ đến tệp **CHAY_UNG_DUNG.bat** trong thư mục của bạn.
7. Tại ô Start in, dán đường dẫn thư mục chứa ứng dụng của bạn.
8. Bấm **Finish** để hoàn tất.

---

## 7. XỬ LÝ SỰ CỐ THƯỜNG GẶP

* **Lỗi thông báo thiếu tệp c10.dll:** Do máy tính chưa cài gói Microsoft Visual C++ Redistributable. Bạn chỉ cần tải và cài đặt tệp vc_redist.x64.exe theo hướng dẫn ở Bước 2.1.
* **Cửa sổ chạy bị dừng lại:** Nếu bạn vô tình bấm chuột vào nền đen của cửa sổ làm xuất hiện chữ Select trên thanh tiêu đề, bạn chỉ cần nhấn phím **Enter** trên bàn phím để tiến trình tiếp tục chạy.
* **Khởi động ứng dụng hàng ngày:** Chỉ cần mở thư mục và nhấp đúp vào tệp **CHAY_UNG_DUNG.bat**.

---
*Tài liệu hướng dẫn cấu hình và triển khai hệ thống Oil Forecast Hub.*

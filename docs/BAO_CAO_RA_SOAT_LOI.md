# Báo cáo rà soát lỗi — Oil Forecast Automated Evaluation Hub

> Ghi lại toàn bộ quá trình rà soát code (backend + frontend), tổng hợp từ phiên làm việc rà soát trực tiếp trên code và trên app đang chạy, đối chiếu với báo cáo của Gemini để xác minh lại từng claim.
>
> Phạm vi: `app_main.py`, `frontend_mockup.html`, `train_all_horizons.py`. Ứng dụng chạy local trong LAN — các lỗi liệt kê ở đây là lỗi **chức năng/logic**, không phải lỗi bảo mật (bảo mật không nằm trong phạm vi rà soát theo yêu cầu).

---

## Phần 1 — Đã sửa xong (12 lỗi, đã test trực tiếp trên app)

| # | Lỗi | Vị trí | Cách sửa |
|---|-----|--------|----------|
| 1 | Overlay "Chào mừng"/tour không làm tối sidebar vì z-index thấp hơn sidebar của Streamlit | JS tour engine trong `app_main.py` | Nâng z-index của lớp phủ, hộp thoại, mũi tên lên rất cao (2 tỷ) để luôn nằm trên mọi phần tử Streamlit |
| 2 | Rò rỉ `addEventListener` (click/scroll/resize) chồng chất mỗi lần Streamlit rerun | Tour engine JS | Đổi sang cơ chế mỗi lần chạy đều gỡ listener cũ rồi gắn listener mới |
| 3 | Không có khóa Job Huấn luyện → nhiều người dùng LAN có thể chạy chồng, ghi đè checkpoint | `app_main.py` (trang Huấn luyện) | Thêm file khóa (`.training.lock`) kiểm tra trước khi cho phép bấm nút bắt đầu job |
| 4 | Card "Dữ liệu hợp lệ / Kiểm tra độ mới / Phạm vi dự báo" 3 cột quá hẹp, chữ rớt dòng xấu | Trang Dự báo | Đổi sang layout 1 cột, icon+nhãn+mô tả trên cùng dòng |
| 5 | Dropdown "Chế độ huấn luyện" bị cắt chữ (`...`) | Trang Huấn luyện | Đổi tỷ lệ cột từ `st.columns(2)` → `st.columns([2,1])` |
| 6 | Trùng `id="training"` trong `frontend_mockup.html` khiến bản UI mới (tabs, mode-help) không bao giờ hiện | `frontend_mockup.html` | Xoá section cũ trùng lặp |
| 7 | Xung đột CSS sidebar: khối "Midnight Navy" cũ còn sót lại đè lên khối sidebar sáng mới → hover ra chữ tối trên nền tối, không đọc được | CSS đầu `app_main.py` | Xoá hẳn khối CSS Midnight Navy cũ, chỉ giữ 1 bộ style sidebar sáng |
| 8 | Multiselect "Chọn mốc dự báo" không lọc thật — luôn tính đủ 7 mốc dù bỏ chọn bớt | `show_live_forecasts()` | Đổi vòng lặp dùng `sel_horizons` (tham số) thay vì hằng số `HORIZONS` |
| 9 | Có thể crash trang "Lịch sử huấn luyện" nếu thiếu `total_rows` (format `,` trên chuỗi `"-"`) | Trang Huấn luyện — tab Lịch sử | Kiểm tra kiểu dữ liệu trước khi format, mặc định hiện "Chưa rõ" |
| 10 | `CUTOFF_DATE` hardcode cứng ngày `2025-09-20`, không tự cập nhật theo thời gian | Hằng số cấu hình | Tính động = (ngày mới nhất trong dữ liệu) − 365 ngày |
| 11 | Bấm thẻ hướng dẫn trên trang "Hướng dẫn sử dụng" chỉ chuyển trang, không mở tour | Tour engine JS | Nguyên nhân: mỗi lần đổi trang Streamlit hủy khung ẩn (iframe) cũ, listener cũ trở thành "listener chết". Sửa: luôn gỡ + gắn lại listener mới mỗi lần chạy |
| 12 | Favicon tab trình duyệt hiển thị sai (emoji 🛢️ không có glyph trên một số máy) | `st.set_page_config` | Tạo `assets/favicon.png` riêng, dùng file ảnh thay vì emoji |

---

## Phần 2 — Đã xác nhận đúng bằng code, CHƯA sửa (8 lỗi)

> Các lỗi này do Gemini phát hiện, đã được tôi tự đọc lại code để xác minh (không dùng agent), có trích dẫn vị trí chính xác.
>
> **Cập nhật:** #13, #14, #15, #16, #17, #18, #19, #24 **đã được sửa và test trực tiếp** (chi tiết trong `LOG_TRIEN_KHAI_A_DEN_D.md`). #20 sau khi đọc kỹ hoá ra **không phải lỗi thật** (đã rút lại). #22 (ở Phần 3 Gemini bên dưới) cũng đã sửa luôn vì phát hiện là nguyên nhân thật đứng sau #20.

| # | Lỗi | Vị trí | Mức độ | Chi tiết / Hậu quả | Trạng thái |
|---|-----|--------|--------|---------------------|:---:|
| 13 | Biến `sel_horizons` không tồn tại trong hàm `run_upload_simulation` (hàm không nhận tham số này, cũng không phải biến global) | `app_main.py` (hàm `run_upload_simulation`) | 🔴 Nghiêm trọng | Ném `NameError` mỗi khi có dữ liệu mới cần backtest; lỗi bị nuốt bởi `except Exception` bao quanh → toàn bộ tính năng "Đánh giá mô hình" luôn thất bại âm thầm | ✅ Đã sửa — test thật: từ rỗng → ra 1276 điểm dữ liệu |
| 14 | `MODEL_DEFS` chỉ khai báo "GUMNet", thiếu "HybridTriNet" | `app_main.py` (hằng số `MODEL_DEFS`) | 🔴 Nghiêm trọng | Chọn "HybridTriNet" ở sidebar không bao giờ ra kết quả dự báo (bị `try/except` trong `load_model()` nuốt lỗi, không crash nhưng cũng không báo gì) | ✅ Đã sửa — test thật: HybridTriNet ra đủ bảng dự báo + backtest 916 điểm |
| 15 | Mũi tên hướng dẫn (tour) tính sai vị trí vì dùng `window.innerWidth`/`innerHeight` của khung ẩn 0×0 thay vì màn hình thật | `app_main.py` (hàm `positionDialogAndArrow`) | 🟡 Trung bình | Công thức luôn cho ra kết quả cố định ~16px bất kể vị trí thật của phần tử được trỏ tới | ✅ Đã sửa (Việc D) — test thật bằng ảnh chụp + job thật |
| 16 | Khóa Job huấn luyện lưu `os.getpid()` — PID của Streamlit server, không phải PID tiến trình huấn luyện con | `app_main.py` (hàm `acquire_training_lock`) | 🟡 Trung bình | `_pid_alive()` kiểm tra nhầm đối tượng | ✅ Đã sửa — ghi đè lại bằng `process.pid` thật ngay sau khi `Popen` chạy |
| 17 | `process.terminate()` có thể cắt ngang tiến trình con đúng lúc đang ghi checkpoint, nếu người dùng đổi trang giữa lúc job chạy | `app_main.py` (khối `finally` của job huấn luyện) | 🟡 Trung bình | Đánh đổi phát sinh từ lần sửa lỗi #3 (khóa job) trước đó | ✅ Đã sửa — `train_all_horizons.py` tự quản lý lock của chính nó (`atexit`), `app_main.py` không còn tự ý `terminate()` nữa |
| 18 | Thiếu nút tải về cho 2 file PDF hướng dẫn có sẵn trong project | Trang "Hướng dẫn sử dụng" | 🟢 Nhẹ | Có file nhưng không cách nào tải qua UI | ✅ Đã sửa — test thật: 2 nút tải hiện đúng |
| 19 | Xáo trộn thứ tự cột đặc trưng khi file dữ liệu thiếu 1 cột ở giữa danh sách | `predict_from_df()` | 🟡 Trung bình | Đệm số 0 vào cuối mảng thay vì đúng vị trí ban đầu → dự báo sai lệch toán học | ✅ Đã sửa bằng `df.reindex(columns=f_cols, fill_value=0.0)` — đã đọc kỹ code, chưa có file lỗi thật để test trực tiếp |
| ~~20~~ | ~~`pct_imp = diff / l1` không có epsilon chống chia cho 0~~ | `app_main.py` | — | **RÚT LẠI: không phải lỗi thật.** Đã có `if l1 and l2:` chặn từ trước (`0.0` là falsy trong Python) | ❌ Không phải lỗi |
| 24 | Tour "rọi sáng" (spotlight) trỏ nhầm phần tử trên trang "Đánh giá mô hình" | `app_main.py` (bước tour "metrics") | 🟡 Trung bình | Selector liệt kê chung chung khiến `querySelector` khớp nhầm phần tử khác trên trang (kể cả trong sidebar) | ✅ Đã sửa (Việc D) — thêm `pickBestElement()` thử từng selector theo thứ tự, chỉ tìm trong vùng nội dung chính |

**Lưu ý về bối cảnh LAN:** vì app chạy chung 1 server cho nhiều người dùng trong LAN, lỗi #16 và #17 (liên quan tới khóa job huấn luyện dùng chung) thực ra **quan trọng hơn** chứ không giảm nhẹ, vì nhiều người có thể cùng lúc thao tác trên cùng 1 phiên bản server.

---

## Phần 3 — Gemini nêu (3 claim, trong `train_all_horizons.py`)

| # | Claim | Trạng thái |
|---|-------|-----------|
| 21 | Nuốt lỗi (`except: entries = []`) khi đọc/ghi `training_history.json`, có thể mất trắng lịch sử các phiên huấn luyện trước nếu bị Windows file-lock đúng lúc | ✅ **Xác nhận đúng, đã sửa** — không còn reset về `[]` rồi ghi đè nữa, nếu đọc lỗi thì bỏ qua lưu phiên đó, không đụng tới lịch sử cũ |
| 22 | Phiên huấn luyện mới không ghi field `avg_val_loss` / `status` vào `training_history.json` | ✅ **Xác nhận đúng, đã sửa** — đây chính là nguyên nhân thật đứng sau claim #20 (KPI Benchmarking/Val Loss TB luôn trống với dữ liệu thật). Đã thêm tính `avg_val_loss` (trung bình val loss các mốc) và `status: "success"` khi lưu entry |
| 23 | Lệch chuỗi log: `app_main.py` tìm `"Best Val Loss"` nhưng `train_all_horizons.py` (nhánh HybridTriNet) in ra `"best_val="` → thanh tiến trình đứng im 0% khi huấn luyện HybridTriNet | ✅ **Xác nhận đúng, đã sửa** — khi làm Việc B (bảng tiến độ theo mốc) đã đổi sang regex chấp nhận cả 2 kiểu chuỗi log, tự động giải quyết luôn claim này |

---

## Phần 4 — Xác minh báo cáo dán lần 2 (claim về `interpolate`/checkpoint)

Người dùng dán thêm 1 báo cáo khác (không rõ nguồn) nêu lỗi `DataFrame cannot interpolate with object dtype` và cảnh báo `"Không thể nạp checkpoint... sẽ học mới"`, kèm trích dẫn file log `task-2082.log`/`task-2087.log`. Đã tự tay kiểm chứng lại bằng `git blame`/grep trực tiếp trên `train_all_horizons.py`:

| Claim | Kết luận |
|---|---|
| Lỗi `interpolate()` trên toàn bộ DataFrame (kể cả cột chữ) gây `TypeError` | ⚠️ **Từng có thật** ở commit cũ `72b8d186` (chưa lọc cột số trước khi nội suy), nhưng **đã được sửa** ở commit `87d456e8` (08/09/2026 — chính là bản đang chạy hiện tại): đã thêm `select_dtypes(include=[np.number])` trước khi `.interpolate()` ở `train_all_horizons.py:101-102`. **Không còn là lỗi đang tồn tại.** |
| Cảnh báo `"⚠️ Không thể nạp checkpoint GUMNet h1, sẽ học mới..."` | ❌ **Không khớp code thật.** Grep toàn bộ `.py` trong project không tìm thấy chuỗi này ở đâu. Message thật trong code là `"🌱 Khởi tạo mô hình GUMNet h{horizon} để tối ưu hóa mới..."` (`train_all_horizons.py:173`) — khác hẳn câu chữ được trích dẫn. |
| File `task-2082.log`, `task-2087.log` được trích dẫn làm bằng chứng | ❌ **Không tồn tại** trong thư mục project (`D:/Anh_Thuy`). |

→ **Kết luận: báo cáo dán lần 2 không đáng tin cậy** (1 phần lỗi thời đã fix, 1 phần không khớp source thật, có thể lấy nhầm từ phiên làm việc/dự án khác). Không dùng làm căn cứ để sửa code.

---

## Phần 5 — Kế hoạch đề xuất (tính năng mới, CHƯA triển khai — chờ duyệt)

Từ ảnh chụp thực tế người dùng gửi (xác nhận đúng lỗi #15 và #24 ở trên) và trao đổi thêm về trải nghiệm khi dùng CPU, thống nhất 3 hạng mục sau — đây là **tính năng mới / cải tiến UX**, không phải sửa lỗi, nên tách riêng khỏi Phần 1/2:

| # | Đề xuất | Mô tả | Trạng thái |
|---|---------|-------|-----------|
| A | Banner gợi ý khi phát hiện dữ liệu mới | Khi upload file: nếu ngày lớn nhất trong file **không mới hơn** dữ liệu hệ thống đang có → hiện bình thường như hiện tại, không cần thêm gì. Nếu **mới hơn** → chủ động hiện banner gợi ý dạng: *"Phát hiện dữ liệu mới đến ngày X — bạn có thể xem thử dự báo ngay, hoặc huấn luyện lại (Finetune) để mô hình chính xác hơn"*, kèm nút bấm nhanh sang trang Huấn luyện. Không bắt buộc, người dùng tự quyết định. | ⏳ Chưa làm |
| B | Hiển thị tiến độ huấn luyện CPU trực quan theo từng mốc | Thay vì chỉ 1 thanh % chung chung, hiện trạng thái riêng từng mốc horizon: `⚪ Đang chờ` / `⏳ Đang chạy (epoch x/y)` / `✅ Đã xong (val loss: ...)` — giúp người dùng biết rõ CPU đang chạy tới đâu, tránh cảm giác "treo máy" khi CPU chạy chậm. | ⏳ Chưa làm |
| C | Khóa/chỉnh UX tab Huấn luyện để tránh người dùng thao tác gây crash | Ngoài khóa file đã có (lỗi #3, đã sửa), cần rà soát thêm UI: vô hiệu hoá các control khác (đổi model, đổi mốc, nút khác) trong lúc job đang chạy để người dùng không vô tình bấm thao tác gây xung đột / góp phần kích hoạt lỗi #17 (`process.terminate()` cắt ngang lúc đang ghi checkpoint). | ⏳ Chưa làm |
| D | Sửa đúng gốc lỗi #15 (mũi tên) và #24 (rọi sáng sai chỗ) | Dùng `window.parent.innerWidth/innerHeight` (màn hình thật) thay vì của khung ẩn 0×0; sửa selector từng bước tour trỏ đúng chính xác 1 phần tử duy nhất thay vì liệt kê chung chung dễ trúng nhầm. | ⏳ Chưa làm |

---

## Tổng kết

- **23 lỗi đã xác nhận chắc chắn bằng code thật, TẤT CẢ đã được sửa và hầu hết đã test trực tiếp trên app** (12 lỗi ban đầu + #13,14,15,16,17,18,19,21,22,23,24 — chi tiết xem `LOG_TRIEN_KHAI_A_DEN_D.md`).
- **#20 đã rút lại** — không phải lỗi thật, đã có sẵn guard chặn chia cho 0.
- **Báo cáo dán lần 2** (claim về `interpolate`/checkpoint) đã kiểm chứng: **không đáng tin cậy** — xem chi tiết Phần 4.
- Báo cáo gốc của Gemini có 2 điểm **lỗi thời/không còn đúng** tại thời điểm rà soát: favicon (đã sửa từ trước), và cơ chế tour không mở lại sau khi chuyển trang (đã sửa, nguyên nhân gốc do "listener chết" theo iframe cũ chứ không hẳn do `sessionStorage`).
- **4 hạng mục A-D (Phần 5) đã triển khai xong** — D/B/C đã test trực tiếp bằng job thật + DevTools; A đã code xong, đang chờ người dùng tự test upload (giới hạn công cụ, không tự động hoá được hộp thoại chọn file OS).
- Chưa có commit/push nào lên nhánh `main` trên GitHub trong toàn bộ quá trình rà soát và sửa lỗi này.

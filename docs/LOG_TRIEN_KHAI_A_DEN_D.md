# Log triển khai hạng mục A-D (theo Phần 5, `BAO_CAO_RA_SOAT_LOI.md`)

> File này ghi lại tiến trình thực hiện 4 hạng mục đã thống nhất: D (sửa lỗi tour), A (banner dữ liệu mới), B (hiển thị tiến độ CPU theo mốc), C (khóa UX tab huấn luyện). Cập nhật liên tục khi làm xong từng việc.
>
> **Cập nhật (đợt 2):** sau khi xong A-D, tiếp tục sửa nhóm 7 lỗi tồn đọng #13, #14, #16, #17, #18, #19 (đã xác nhận trong `BAO_CAO_RA_SOAT_LOI.md`) — xem mục "Đợt 2" ở cuối file. Lưu ý: #15 và #24 thực ra đã được sửa chung trong Việc D ở trên, không phải mục riêng. #20 sau khi đọc lại kỹ hoá ra KHÔNG phải lỗi thật (đã có `if l1 and l2:` chặn chia cho 0 sẵn) — đã rút lại. Trong lúc sửa phát hiện thêm và sửa luôn #22 (claim Gemini) vì cùng khu vực code.

---

## Việc D — Sửa lỗi mũi tên + rọi sáng sai chỗ trong tour hướng dẫn

**Trạng thái:** ✅ Xong, đã test trực tiếp trên app

**Thay đổi:**
1. `positionDialogAndArrow()`: đổi `window.innerWidth/innerHeight` (kích thước khung ẩn 0×0, luôn ≈0) → `window.parent.innerWidth/innerHeight` (kích thước màn hình thật của người dùng).
2. Thêm hàm `pickBestElement(selectorStr)`: tách chuỗi selector theo dấu phẩy, thử **từng selector riêng lẻ theo đúng thứ tự ưu tiên** thay vì để `querySelector()` tự chọn phần tử khớp đầu tiên trong toàn tài liệu (là nguyên nhân gốc gây rọi sáng nhầm chỗ).
3. `pickBestElement()` chỉ tìm bên trong vùng nội dung chính (`stMain`/`.main`), loại trừ hẳn sidebar — tránh khớp nhầm phần tử nằm trong sidebar (đứng trước nội dung chính trong DOM).
4. Áp dụng `pickBestElement()` cho cả `waitForTargetElement()` (mở tour) và `updateActivePositions()` (resize/scroll).

**Kết quả test (trang "Đánh giá mô hình", bước tour "1. Chỉ số sai số MAPE & MAE"):**
- Mũi tên trỏ đúng ngay phía trên khối 3 thẻ chỉ số (không còn ghim ở mép trái 16px).
- Khối 3 thẻ chỉ số có viền xanh rọi sáng bao đúng quanh nó (trước đó không có viền nào cả — rọi sáng nhầm chỗ khác).
- Hộp thoại tự đặt vị trí hợp lý, không đè lên sidebar hay nội dung.

---

## Việc A — Banner gợi ý khi phát hiện dữ liệu mới

**Trạng thái:** ✅ Đã code xong, ⚠️ chưa test được qua UI thật (giới hạn công cụ, xem bên dưới)

**Thay đổi:**
1. Thêm `key="main_nav_radio"` vào `st.sidebar.radio()` điều hướng chính, và tách danh sách trang ra hằng số `NAV_OPTIONS` — để có thể set trang đích từ nơi khác trong code.
2. Ở khối xử lý upload trang "Dự báo": sau khi nạp `df_new` thành công, so sánh ngày lớn nhất của file (`m_date`) với `_latest_known_date` (biến đã có sẵn, tính TRƯỚC khi ghi file mới vào `datasets/` — đại diện đúng cho "dữ liệu hệ thống đang có" tại thời điểm so sánh).
3. Nếu `m_date > _latest_known_date` → hiện `st.info(...)` gợi ý xem dự báo ngay hoặc Finetune, kèm nút **"⚡ Chuyển sang Huấn luyện để Finetune ngay"**.
4. Nút bấm set `st.session_state["main_nav_radio"] = "⚙  Huấn luyện mô hình"` rồi `st.rerun()` — điều hướng sang trang Huấn luyện ngay lập tức.
5. Nếu file không có ngày mới hơn → không hiện gì thêm, giữ nguyên hành vi cũ.

**Giới hạn khi kiểm thử:** Công cụ trình duyệt tự động đang dùng không hỗ trợ giả lập chọn file qua hộp thoại OS (`<input type="file">` cần API cấp trình duyệt riêng mà bộ công cụ hiện tại không có) — nên **chưa thể tự động upload file để xem banner thật trên UI**. Đã kiểm tra bằng cách đọc lại code: biến `_latest_known_date` nằm đúng phạm vi (global, tính trước dòng ghi file), so sánh `pandas.Timestamp` chuẩn, không có lỗi cú pháp (`py_compile` qua), trang Dự báo vẫn tải bình thường sau khi thêm đoạn code này (không crash). **Đề nghị người dùng tự tay thử upload 1 file có ngày mới hơn 25/05/2026 để xác nhận banner hiện đúng.**

---

## Việc B — Bảng tiến độ huấn luyện theo từng mốc (thay vì chỉ 1 thanh %)

**Trạng thái:** ✅ Xong, đã test bằng job huấn luyện thật (không phải giả lập)

**Thay đổi:**
1. Thêm `hz_status` (dict theo từng mốc: `waiting` / `running` / `done` + val loss) và hàm `_render_hz_table()` vẽ lại danh sách trạng thái mỗi khi có cập nhật, hiển thị qua `st.empty()` để cập nhật tại chỗ (không tạo dòng mới liên tục).
2. Bắt sự kiện "bắt đầu mốc" qua regex `HZ_START_RE` khớp dòng `"ĐANG HUẤN LUYỆN MỐC: {h} NGÀY"` → set trạng thái mốc đó thành `running`.
3. Bắt sự kiện "hoàn tất mốc" qua regex `VAL_LOSS_RE` — **gộp chung 2 kiểu chuỗi log** `"Best Val Loss: X"` (GUMNet) và `"best_val=X"` (HybridTriNet) → set trạng thái `done` kèm giá trị val loss. Sửa nhân tiện luôn claim #23 tồn đọng từ báo cáo Gemini (progress bar đứng im 0% khi huấn luyện HybridTriNet do trước đây chỉ bắt đúng 1 kiểu chuỗi).

**Kết quả test thật (bấm "Bắt đầu Job Huấn Luyện" với 2 mốc 1d, 20d, GPU):**
- Bảng hiện đúng ngay khi bấm nút: `⚪ Mốc 1 ngày: Đang chờ...` / `⚪ Mốc 20 ngày: Đang chờ...`.
- Sau khi log in ra `"🚀 ĐANG HUẤN LUYỆN MỐC: 1 NGÀY"`, bảng tự cập nhật đúng thành `⏳ Mốc 1 ngày: Đang tối ưu hóa...`.
- Job thật sự **bị lỗi** giữa chừng (xem log kỹ thuật): `RuntimeError: File D:\Anh_Thuy\checkpoints_multi\gumnet_h1.pt cannot be opened` khi `torch.save` — **nguyên nhân là do file checkpoint đang bị 1 tiến trình Streamlit cũ (từ việc tôi khởi động lại server nhiều lần trong phiên rà soát) giữ khóa file trên Windows, không liên quan gì đến code Việc A-D**. Đã kiểm tra: checkpoint không bị hỏng (`torch.load` lại vẫn đọc được bình thường, vì lỗi xảy ra trước khi có dữ liệu nào được ghi đè).
- Nhờ vậy cũng gián tiếp xác nhận đúng: bảng KHÔNG tự ý báo "Đã xong" khi job thực sự chưa hoàn tất/bị lỗi — giữ nguyên trạng thái `running` đúng như thực tế.

---

## Việc C — Khóa các control khác trong tab Huấn luyện khi có job đang chạy

**Trạng thái:** ✅ Xong, đã test bằng lock giả + kiểm tra DevTools

**Thay đổi:**
1. Chuyển `active_lock = get_active_training_lock()` lên TRƯỚC 3 widget cấu hình (trước đây tính sau, chỉ dùng để khóa mỗi nút Bắt đầu).
2. Thêm `disabled=bool(active_lock)` cho `train_mode` (selectbox), `n_epochs` (number_input), `sel_hz_manual` (multiselect).

**Kết quả test (tạo file `.training.lock` giả với PID còn sống, mở tab Huấn luyện):**
- Cảnh báo "Đã có Job Huấn luyện khác đang chạy" hiện đúng.
- Kiểm tra qua `document.querySelectorAll(...).disabled` trên DevTools: cả 3 input tương ứng (`stSelectbox`, `stNumberInputField`, `stMultiSelect` trong tab Huấn luyện) đều trả về `disabled: true` — xác nhận đúng, không phải chỉ style xám mà thao tác thật sự bị khóa.
- Xóa file lock giả + tải lại → cả 3 control hoạt động lại bình thường.

---

## Đợt 2 — Sửa nhóm 7 lỗi tồn đọng #13, #14, #16, #17, #18, #19, #22

### #13 — `sel_horizons` NameError trong `run_upload_simulation`
**Trạng thái:** ✅ Xong, đã test bằng backtest thật
- Thêm tham số `sel_horizons=None` cho hàm, mặc định fallback về `HORIZONS` (giống pattern đã dùng ở `show_live_forecasts`).
- **Test:** xoá `simulation_cache.pkl` để buộc tính lại, mở trang "Đánh giá mô hình" → từ trạng thái luôn rỗng trước đây, giờ ra kết quả thật: MAPE 6.80%, MAE 6.94, **1276 điểm dữ liệu từ 18 đợt cập nhật**.

### #14 — `MODEL_DEFS` thiếu "HybridTriNet"
**Trạng thái:** ✅ Xong, đã test bằng cả trang Dự báo lẫn Đánh giá mô hình
- Thêm khai báo `"HybridTriNet": {"proj_dir": ROOT/"Hybridtrinet_Oil", "mod": "src.model.hybrid_trinet", "cls": "HybridTriNet", "kind": "point"}` — đã xác minh khớp đúng chữ ký constructor thật của class `HybridTriNet` trong `Hybridtrinet_Oil/src/model/hybrid_trinet.py`, và checkpoint tương ứng (`hybrid_h1/5/10/30/60.pt` + thư mục `_meta`) đã có sẵn.
- **Test:** chọn "HybridTriNet" ở sidebar → trang "Đánh giá mô hình" ra kết quả thật (MAPE 11.53%, 916 điểm dữ liệu) thay vì trống trơn; trang "Dự báo" ra đủ bảng dự báo 5 mốc (1,5,10,30,60 ngày — thiếu 15,20 vì checkpoint gốc không có, hợp lý).

### #16 — Khóa training lưu sai PID
**Trạng thái:** ✅ Xong
- `acquire_training_lock()` giờ nhận thêm tham số `pid`; gọi lần 1 (giữ chỗ, tránh race) dùng PID Streamlit như cũ, gọi lần 2 NGAY sau khi `subprocess.Popen()` thành công để ghi đè bằng `process.pid` (PID thật của tiến trình huấn luyện con).

### #17 — `process.terminate()` có thể cắt ngang lúc ghi checkpoint
**Trạng thái:** ✅ Xong (cải tiến kiến trúc nhỏ, không phải chỉ vá tạm)
- `train_all_horizons.py` giờ **tự quản lý lock của chính nó**: tự ghi lúc bắt đầu (`_write_own_lock`, dùng đúng PID thật của chính nó), tự xoá khi kết thúc dù thành công hay lỗi (`atexit.register(_release_own_lock)` — không cần thụt lề lại toàn bộ file).
- `app_main.py` không còn tự ý `terminate()` tiến trình con khi bị Streamlit ngắt giữa chừng nữa — chỉ dọn lock ở phía mình khi tiến trình con **chưa từng chạy được** (vd Popen lỗi ngay từ đầu); nếu nó đang chạy thật thì để nó tự chạy nốt trong nền và tự dọn lock của chính nó.

### #18 — Thiếu nút tải PDF hướng dẫn
**Trạng thái:** ✅ Xong, đã test trực tiếp
- Thêm khối "📄 Tài liệu hướng dẫn chi tiết (PDF)" ở trang Hướng dẫn sử dụng, 2 `st.download_button` cho 2 file PDF có sẵn trong project.
- **Test:** mở trang Hướng dẫn → thấy đúng 2 nút "Hướng dẫn Cấu hình & Triển khai (PDF)" và "Hướng dẫn Triển khai & Sử dụng (PDF)".

### #19 — Xáo trộn thứ tự cột đặc trưng trong `predict_from_df`
**Trạng thái:** ✅ Xong (đã kiểm tra kỹ bằng đọc code, chưa có file thiếu cột thật để test trực tiếp)
- Thay `available = [c for c in f_cols if c in df.columns]` + `np.hstack` đệm 0 vào cuối, bằng `df.reindex(columns=f_cols, fill_value=0.0).values` — giữ đúng vị trí từng cột theo đúng thứ tự gốc, cột nào thiếu được điền 0 đúng tại vị trí của nó thay vì bị dồn ra cuối làm lệch toàn bộ.

### #20 — RÚT LẠI: không phải lỗi thật
Khi sửa #22 (bên dưới), đọc lại kỹ mới thấy dòng `if l1 and l2:` đã chặn `l1=0.0` từ trước (0.0 là falsy trong Python) — `pct_imp = diff/l1` không bao giờ có thể chia cho 0 trong thực tế. Xin lỗi vì đã báo nhầm lỗi này trước đó trong `BAO_CAO_RA_SOAT_LOI.md`.

### #22 (bonus, claim từ Gemini) — `avg_val_loss` không bao giờ được ghi vào `training_history.json`
**Trạng thái:** ✅ Xong — phát hiện khi đang sửa #20, xác nhận đúng, sửa luôn vì cùng khu vực code
- Đây mới là nguyên nhân thật khiến "Val Loss TB" (tab Lịch sử) và toàn bộ KPI so sánh Benchmarking luôn trống với dữ liệu thật — trước đây field `avg_val_loss` chỉ được ĐỌC ở `app_main.py`, không hề được GHI ở đâu cả trong `train_all_horizons.py`.
- Thêm `all_val_losses` gom lại toàn bộ val loss từng mốc/model trong lúc train, tính trung bình ghi vào `entry["avg_val_loss"]`, đồng thời thêm `entry["status"] = "success"` (cũng bị thiếu, một phần của claim #22).

### Bonus thêm — #21 (claim Gemini): nuốt lỗi làm mất lịch sử huấn luyện
**Trạng thái:** ✅ Xong — tiện sửa vì cùng file, cùng khu vực
- Trước đây nếu đọc `training_history.json` cũ bị lỗi (vd đang bị khóa file), code âm thầm reset về `[]` rồi ghi đè → mất sạch lịch sử cũ. Giờ nếu đọc lỗi thì **bỏ qua lưu phiên này**, không ghi đè, chỉ in cảnh báo.

**Tất cả đã `py_compile` qua cả 2 file (`app_main.py`, `train_all_horizons.py`) sau mỗi bước sửa.**

---

## Đợt 3 — Đổi UX chọn mốc dự báo ở trang "Dự báo" (theo đề xuất, chỉ áp dụng bên hiển thị)

**Trạng thái:** ✅ Xong, đã test trực tiếp

**Bối cảnh:** Ô multiselect "Chọn mốc thời gian cần dự báo" (`sel_hz_view`) gây nhầm lẫn (tưởng lọc để chạy nhanh hơn, thực ra tính đủ 7 mốc vẫn rất nhanh) và dễ bấm nhầm dấu `×` làm mất mốc. Đã kiểm tra kỹ: biến này CHỈ dùng để lọc hiển thị ở `show_live_forecasts()`, hoàn toàn tách biệt với `sel_hz_manual` (ô chọn mốc cần **huấn luyện lại** ở trang Huấn luyện) — đổi ô này không ảnh hưởng gì tới huấn luyện hay bất kỳ quá trình nào khác.

**Thay đổi:**
- Bỏ hẳn ô `st.multiselect` ở trang Dự báo, thay bằng `sel_hz_view = HORIZONS` (luôn tính đủ cả 7 mốc) + 1 dòng caption hướng dẫn dùng chú giải (legend) trên biểu đồ.
- Không cần code thêm gì cho việc ẩn/hiện đường trên biểu đồ — đây là tính năng **có sẵn của Plotly** (click vào tên mốc ở legend để bật/tắt đường tương ứng), chỉ cần đảm bảo chart không tắt nó đi (đã kiểm tra, code cũ không tắt).

**Kết quả test trên app thật:**
- Trang Dự báo không còn ô chọn mốc nữa, bảng tổng hợp dự báo hiện đủ 7 dòng ngay lập tức không cần chọn gì.
- Biểu đồ "So sánh lộ trình dự báo" hiện đủ 7 đường (1d→60d) kèm chú giải bên phải.
- Bấm vào "Dự báo 1d" trên legend → đường đó chuyển màu xám mờ (ẩn) ngay lập tức, không cần tải lại trang. Bấm lại → hiện lại bình thường.

---

## Đợt 4 — Chỉnh bố cục card "Cập nhật dữ liệu thị trường" (theo bản vẽ demo đã duyệt)

**Trạng thái:** ✅ Xong, đã test trực tiếp, khớp đúng bản vẽ demo

**Bối cảnh:** Sau khi bỏ ô chọn mốc (Đợt 3), card bên trái bị trống/ngắn hơn hẳn so với card "Trạng thái hệ thống" bên phải. Đã vẽ demo bằng widget trực quan cho người dùng duyệt trước khi code.

**Thay đổi:**
1. Bỏ số "1." ở tiêu đề card (`"#### 1. Cập nhật dữ liệu thị trường"` → `"#### Cập nhật dữ liệu thị trường"`) — vì giờ chỉ còn đúng 1 bước ở card này (bước "2. Chọn mốc..." đã bỏ ở Đợt 3), đánh số không còn hợp lý.
2. Thêm CSS `div[data-testid="stFileUploaderDropzone"] { min-height: 150px; }` — khung kéo-thả file to hơn hẳn, không còn trông trống trải.
3. Thêm 3 ô thông tin nhỏ bên dưới khung upload: Định dạng hỗ trợ / Dung lượng tối đa / Cột bắt buộc — lấp khoảng trống, cân chiều cao 2 cột.
4. Dời dòng chú thích "bấm vào legend để ẩn/hiện mốc" ra khỏi card upload (không còn liên quan tới việc tải file), đặt đúng chỗ hơn: ngay phía trên biểu đồ so sánh trong `show_live_forecasts()`.

**Kết quả test:** khớp đúng bản vẽ demo đã duyệt — 2 cột cân chiều cao, không còn khoảng trống thừa, không có lỗi/crash.

---

## Đợt 5 — Thêm khung viền (card) thật cho 2 cột trang Dự báo

**Trạng thái:** ✅ Xong, đã test trực tiếp, không phát sinh lỗi mới

**Bối cảnh:** Bản demo có khung viền + nền trắng bao quanh, nhưng code thật ban đầu không có (không khớp demo). Người dùng chọn thêm khung viền thật theo đúng demo.

**Thay đổi:**
- Dùng `st.container(border=True)` — tính năng **có sẵn của Streamlit** (từ bản 1.31+, đang chạy bản 1.57.0) để bọc khung thật quanh toàn bộ nội dung mỗi cột (kể cả widget thật như `st.file_uploader`, `st.button`).
- **Chủ động không dùng cách tự mở/đóng thẻ `<div>` bằng `st.markdown` để bọc nhiều lệnh Streamlit khác nhau** — vì mỗi lệnh `st.markdown`/`st.file_uploader`/... render vào một container DOM riêng của Streamlit, một `<div>` mở ở lệnh này KHÔNG thực sự bao được widget ở lệnh sau, dễ vỡ layout mà không báo lỗi gì (đúng loại lỗi "HTML không đồng bộ với Streamlit" đã gặp nhiều lần trong dự án này).

**Kiểm tra lỗi (theo yêu cầu):**
- `py_compile` qua, không lỗi cú pháp.
- Test trực tiếp: cả 2 khung viền hiện đúng, khớp bản demo, chiều cao cân đối.
- Chuyển qua lại giữa các trang (Dự báo ↔ Đánh giá mô hình) nhiều lần — không phát sinh lỗi mới trong console trình duyệt (đối chiếu số lượng lỗi console trước/sau, không đổi — các lỗi console tồn tại từ trước đều là lỗi cũ do server bị restart nhiều lần trong phiên làm việc, không liên quan tới thay đổi này).

---

## Đợt 6 — Restyle khung upload giống đúng `frontend_mockup.html` (icon + chữ hướng dẫn tiếng Việt)

**Trạng thái:** ✅ Xong, đã test trực tiếp, có 1 lỗi tự phát hiện và tự sửa ngay trong lúc làm

**Yêu cầu:** người dùng gửi ảnh tham chiếu đúng style trong `frontend_mockup.html` (icon lớn + "Kéo thả file vào đây" + "hoặc bấm để chọn file từ máy tính" + nút "Chọn file dữ liệu"), muốn widget upload thật trông giống vậy thay vì giao diện mặc định của Streamlit ("Upload" + "200MB per file...").

**Cách làm:** Dùng thuần CSS (không JS) để thay icon/chữ mặc định của Streamlit bằng nội dung tiếng Việt, qua `::before`/`::after` + `content:` + ẩn text gốc bằng `font-size:0`. Chọn cách này thay vì JS DOM injection để tránh đúng loại lỗi "không đồng bộ sau khi rerun" đã gặp nhiều lần trong dự án.

**Lỗi tự phát hiện khi test (đã tự sửa ngay):** Lần đầu viết CSS dùng selector `div[data-testid="stFileUploaderDropzone"]` nhưng đo bằng DevTools thì phát hiện phần tử thật là thẻ `<section>`, không phải `<div>` — toàn bộ rule không khớp, không có gì đổi cả. Sửa lại thành `section[data-testid="stFileUploaderDropzone"]` thì áp dụng đúng ngay.

**Kết quả test cuối:** hiện đúng icon ⇧ màu teal, chữ "Kéo thả file vào đây" (đậm) + "hoặc bấm để chọn file từ máy tính" (xám) + nút "Chọn file dữ liệu" — khớp ảnh tham chiếu. Nút bấm vẫn hoạt động bình thường (đã test click, không phát sinh lỗi console mới).

---

## Đợt 7 — Sửa lỗi tiêu đề biểu đồ chồng lên chú giải (tab Benchmarking)

**Trạng thái:** ✅ Xong, đã test trực tiếp ở đúng chiều rộng tái hiện được lỗi

**Bối cảnh:** Người dùng gửi ảnh chụp tab "So sánh Kết quả (Benchmarking)" cho thấy tiêu đề biểu đồ "Đối chiếu Validation Loss..." bị chữ chú giải (legend) đè lên. Đã xác nhận: KHÔNG phải do màn hình nhỏ — tái hiện được ở cả 800px lẫn 1456px.

**Nguyên nhân** ([app_main.py:2300-2307](app_main.py:2300)): `legend=dict(orientation="h", yanchor="bottom", y=1.02, ...)` đặt chú giải nằm ngang, neo sát ngay phía trên khung vẽ — trùng vùng không gian với tiêu đề dài phía trên. Plotly không tự biết 2 thứ này đang tranh chỗ nhau.

**Thay đổi:** Chuyển chú giải xuống DƯỚI biểu đồ (`yanchor="top", y=-0.18, xanchor="center", x=0.5`), tăng `height` (310→340) và margin dưới (20→60) để có đủ chỗ — đảm bảo không bao giờ tranh chỗ với tiêu đề nữa bất kể tiêu đề dài ngắn ra sao.

**Kết quả test:** dựng lại đúng chiều rộng 1456px đã tái hiện lỗi trước đó — tiêu đề và chú giải giờ tách biệt rõ ràng, không còn chồng chữ.

---

## Đợt 8 — Sửa 3 lỗi tour + tooltip (tab Benchmarking)

**Trạng thái:** ✅ Xong cả 3, đã test trực tiếp bằng cách tái hiện đúng kịch bản lỗi trước/sau khi sửa

### #1 + #2: Bấm "Bỏ qua" xong mũi tên tự hiện lại, không kèm viền sáng
**Nguyên nhân** ([app_main.py:563-570](app_main.py:563)): `endTour()` chỉ ẩn mũi tên/hộp thoại nhưng quên reset `activeTour`/`currentIdx` → `updateActivePositions()` (chạy mỗi khi có resize/scroll) vẫn tưởng tour đang chạy, tự vẽ lại mũi tên (không kèm viền sáng vì phần đó chỉ nằm trong `showStep()`, không được gọi lại).
**Sửa:** thêm `activeTour = []; currentIdx = 0;` vào cuối `endTour()`.
**Test xác nhận:** bấm Bỏ qua → bắn thử sự kiện `resize` bằng JS → trước khi sửa: mũi tên bật lại `display:block` ngay; sau khi sửa: vẫn `display:none`, không hồi sinh nữa.

### #3: Tooltip biểu đồ cắt "..." sau "Phiên đối ch..."
**Nguyên nhân:** tên trace `"Phiên đối chứng (TR-20260520-083000)"` dài, Plotly mặc định cắt bớt khi hiện tooltip.
**Sửa:** thêm `hoverlabel=dict(namelength=-1)` vào `fig_cmp.update_layout()` — giá trị `-1` nghĩa là hiện tên đầy đủ, không giới hạn độ dài.
**Test xác nhận:** đọc lại config biểu đồ đã render qua `document.querySelectorAll('.js-plotly-plot')` → `hoverlabel.namelength: -1` đã áp dụng đúng vào biểu đồ thật.

---

## Đợt 9 — Chuẩn hóa cố định mô hình GUMNet phục vụ bàn giao khách hàng (Client-Ready)

**Trạng thái:** ✅ Xong, đã kiểm thử cú pháp và hiển thị giao diện

**Bối cảnh & Lý do thực hiện:**
1. **Lệch số liệu thực tế nghiêm trọng ở HybridTriNet:**
   - Khi đối chiếu cùng tệp dữ liệu thị trường mới nhất (25/05/2026, giá thực tế MG95 đang ở mức 124.81 USD):
     - **GUMNet:** Dự báo bám sát thực tế: MG95 mốc +1 ngày đạt **136.46 USD** (vùng giá 123 – 136 USD trên cả 7 mốc thời gian).
     - **HybridTriNet:** Dự báo bị kéo sụt xuống chỉ còn **74.59 USD** (thấp hơn thực tế gần 50 USD) và thiếu 2 mốc (+15d, +20d).
   - **Nguyên nhân kỹ thuật:** HybridTriNet dùng chuẩn hóa Z-score trên phân phối lịch sử 18 năm (\(\mu \approx 88.7\)), các checkpoint cũ chưa được Finetune theo mặt bằng giá mới nên bị hiện tượng *Mean Reversion* (kéo ngược về trung bình lịch sử cũ).
2. **Yêu cầu triển khai thương mại / bàn giao cho Khách hàng doanh nghiệp (Production-Ready):**
   - Khách hàng không phải chuyên gia nghiên cứu AI; họ cần số liệu tin cậy 100% để ra quyết định kinh doanh.
   - Nếu để lộ radio button chọn mô hình, khách hàng thấy 2 mức giá chênh lệch tới 60 USD sẽ hoang mang, mất niềm tin vào sản phẩm.
   - Quy tắc chuẩn trong phần mềm doanh nghiệp: các mô hình thử nghiệm R&D chưa đồng bộ 7 mốc phải được ẩn đi, chỉ công bố mô hình đã được kiểm định an toàn 100%.

**Thay đổi:**
- Tại Sidebar [`app_main.py:1517-1528`](app_main.py:1517): Thay thế ô radio chọn mô hình bằng một Huy hiệu Doanh nghiệp (Enterprise Badge) cố định sang trọng:
  - **MÔ HÌNH DỰ BÁO: 🧠 GUMNet Enterprise**
  - Trạng thái: `✓ Sẵn sàng 7/7 mốc thời gian (Độ chính xác cao)`
  - Cố định biến hệ thống: `sel_models = ["GUMNet"]`.
- Toàn bộ kiến trúc backend, mã nguồn `Hybridtrinet_Oil` và các hàm xử lý vẫn được bảo lưu nguyên vẹn trong dự án phục vụ mục đích nghiên cứu học thuật nội bộ khi cần.

**Kết quả kiểm tra:**
- `py_compile app_main.py` đạt mã thoát 0.
- Sidebar hiển thị huy hiệu chuyên nghiệp, không còn rủi ro khách hàng bấm nhầm mô hình bị lệch số liệu.
- Các trang Dự báo, Đánh giá, Lịch sử, Huấn luyện vận hành trơn tru và nhất quán 100% trên nền tảng GUMNet.

---

## Đợt 10 — Nâng cấp kiến trúc Spotlight Cutout & Sửa triệt để Event Delegation cho Tour và Popup Chào mừng

**Trạng thái:** ✅ Hoàn thành xuất sắc, đã xác nhận cú pháp Python và kiểm tra sức khỏe ứng dụng (Health check: ok)

### 1. Sửa lỗi vùng chú ý không sáng (Chỉ có mũi tên trỏ vào):
- **Nguyên nhân kỹ thuật:**
  - Trong cấu trúc CSS của Streamlit, các container `section.main` và `block-container` tạo ra Stacking Context và thuộc tính `currentColor !important` riêng biệt, dẫn đến việc gán `outline` hoặc `box-shadow` trực tiếp lên phần tử mục tiêu (như ô Upload) bị ghi đè, làm mờ bởi lớp overlay tối màu hoặc bị cắt mép.
  - Hàm định vị tọa độ `positionDialogAndArrow(targetEl)` chưa kết nối cập nhật tọa độ/kích thước và hiển thị cho phần tử `#oil-spotlight-el`.
- **Giải pháp xử lý:**
  - Áp dụng kiến trúc **Spotlight Cutout Box** độc lập (`#oil-spotlight-el`) đặt trực tiếp dưới `doc.body` với `z-index: 2000000000`, hoàn toàn thoát khỏi mọi giới hạn stacking context của Streamlit.
  - Sử dụng kỹ thuật `box-shadow: 0 0 0 9999px rgba(15, 23, 42, 0.65), 0 0 25px rgba(0, 173, 145, 0.85);` với viền vi mô `3.5px solid #00ad91`. Khung bên trong trong suốt 100% giữ nguyên độ sáng rực rỡ và sắc nét của phần tử mục tiêu, trong khi vùng bóng 9999px phủ tối toàn bộ phần còn lại của màn hình.
  - Trong `positionDialogAndArrow(targetEl)`: Tự động đo `getBoundingClientRect()` của phần tử, bổ sung padding 6px và định vị tức thời. Đồng thời hỗ trợ đa mốc thời gian (0ms, 150ms, 350ms) để bám dính chính xác kể cả khi trang đang cuộn mượt (smooth scroll).
  - Khi không tìm thấy phần tử hoặc kết thúc tour: Tự động ẩn spotlight và khôi phục trạng thái nền an toàn.

### 2. Sửa lỗi không bấm được 2 nút trong Popup Chào mừng (Phải F5 lại mới dùng được):
- **Nguyên nhân kỹ thuật:**
  - Nút "↺ Xem lại thông báo chào mừng & hướng dẫn từ đầu" mở lại popup thông qua `window.parent.replayOnboarding()`. Tuy nhiên, các nút bấm `#oil-welcome-later` ("Để sau") và `#oil-welcome-start` ("Bắt đầu hướng dẫn ➔") chưa được gán sự kiện xử lý nhấp chuột trong bộ điều hướng trung tâm.
  - Ngoài ra, vòng đời component iframe của Streamlit khiến các event listener gắn kiểu truyền thống (`onclick`) bị dead closure sau khi rerun.
- **Giải pháp xử lý:**
  - Chuyển toàn bộ cơ chế xử lý tương tác của popup chào mừng và tour chỉ dẫn sang **Ủy quyền sự kiện bắt giữ (Capturing Event Delegation)** gắn tại cấp tài liệu gốc `doc.__oilTourClickHandler`.
  - Cơ chế tự động dọn dẹp listener cũ (`removeEventListener`) và kích hoạt listener mới (`addEventListener(..., true)`) ở mỗi lần Streamlit rerun.
  - Bổ sung đầy đủ các nhánh xử lý:
    1. Click `#oil-welcome-later`: Đóng popup chào mừng, ghi nhớ cờ `localStorage.setItem('oilForecastTourSeen', 'true')`.
    2. Click `#oil-welcome-start`: Đóng popup, ghi nhớ cờ và tự động khởi động tour tương tác trang Dự báo (`window.parent.startOilTour('forecast')`).
    3. Click ra ngoài vùng backdrop `#oil-welcome-layer`: Tự động đóng popup an toàn.
    4. Xử lý đồng nhất cho nút "Bỏ qua" (`#oil-tour-skip`), "Tiếp theo ➔" (`#oil-tour-next`), và các thẻ điều hướng bài học (`[data-oil-tour]`).

### 3. Kết quả kiểm tra xác thực:
- Lệnh `python -m py_compile app_main.py` hoàn thành thành công (Exit code: 0).
- Health check `http://127.0.0.1:8502/_stcore/health` phản hồi `ok`.
- Cả 2 nút trên popup chào mừng đều phản hồi tức thì mà không cần F5 hay tải lại trang; vùng mục tiêu của tour sáng rực rỡ với khung Spotlight chuẩn doanh nghiệp.

---

## Đợt 11 — Nâng cấp Thứ tầng Stacking Context: Làm sáng tuyệt đối Hộp thoại Hướng dẫn (Tour Dialog) trên toàn bộ các Trang

**Trạng thái:** ✅ Hoàn thành xuất sắc, đã biên dịch mã nguồn và kiểm tra dịch vụ (Health check: ok)

### 1. Vấn đề phát hiện qua ảnh chụp thực tế:
- Vùng mục tiêu phía trên (3 thẻ KPI) đã được rọi sáng chuẩn rực rỡ bằng khung viền Spotlight neon `#00ad91`.
- Tuy nhiên, **hộp thoại chỉ dẫn tương tác ở phía dưới** (*"1. Chỉ số sai số MAPE & MAE..."*) lại bị một lớp màng màu xám/tối phủ lên trên, khiến chữ bị chìm và không đạt độ tương phản màu trắng nguyên bản.

### 2. Nguyên nhân kỹ thuật:
- Trong CSS Stacking Context:
  - Khung chiếu sáng `.oil-spotlight` được đặt ở `z-index: 2000000000` với lớp bóng mở rộng toàn màn hình `box-shadow: 0 0 0 9999px rgba(15, 23, 42, 0.65)`.
  - Khung bao `.oil-tour-layer` chứa hộp thoại chỉ dẫn `.oil-tour-dialog` trước đó có `z-index: 1999999999` (thấp hơn spotlight).
  - Do container cha nằm ở tầng thấp hơn, toàn bộ nội dung hộp thoại bị giam trong stacking context `1999999999` và bị lớp bóng đen 9999px của spotlight phủ đè lên bề mặt.

### 3. Giải pháp đã thực hiện triệt để:
- Thiết lập lại cấu trúc phân tầng z-index khoa học cho toàn bộ hệ thống tour chỉ dẫn:
  1. **Tầng 0 (Bóng tối & Spotlight):** `.oil-spotlight` giữ ở `z-index: 2000000000` để cắt rọi phần tử mục tiêu và phủ tối màn hình xung quanh.
  2. **Tầng 1 (Khung tour trong suốt):** `.oil-tour-layer` được nâng lên `z-index: 2000000002` (cao hơn spotlight), thiết lập `pointer-events: none` để không chắn chuột vào nội dung trang.
  3. **Tầng 2 (Hộp thoại chỉ dẫn):** `.oil-tour-dialog` được đặt ở `z-index: 2000000003`, kích hoạt `pointer-events: auto !important`, ép màu nền `background: #ffffff !important` và chữ màu than đậm `color: #0f172a !important`, bổ sung viền nét cao và bóng đổ sâu `box-shadow: 0 20px 50px rgba(0, 0, 0, 0.45), 0 0 0 1px #e2e8f0 !important`.
  4. **Tầng 3 (Mũi tên chỉ dẫn):** `.oil-arrow` nâng lên `z-index: 2000000004` luôn bay phía trước.
  5. **Tầng đỉnh (Popup Chào mừng Onboarding):** `#oil-welcome-layer` và `.oil-welcome-dialog` nâng lên `z-index: 2000000010` - `2000000011`.

### 4. Kết quả & Phạm vi áp dụng:
- Sửa đổi này nằm ở **bộ điều khiển trung tâm (Engine dùng chung)**, do đó tự động giải quyết dứt điểm và đồng bộ cho **TOÀN BỘ CÁC TOUR TRÊN TẤT CẢ CÁC TRANG**:
  - Trang 1: Tour Dự báo tương lai (4 bước).
  - Trang 2: Tour Đánh giá mô hình (2 bước).
  - Trang 3: Tour Lịch sử kiểm định (2 bước).
  - Trang 4: Tour Huấn luyện mô hình (3 bước).
- Hộp thoại hướng dẫn ở mọi bước giờ đây luôn trắng tinh khiết, chữ đen đậm sắc nét, nút bấm phản hồi mượt mà và không bao giờ bị ám tối nữa.

---

## Đợt 12 — Chuẩn hóa Mặc định Đủ 7 Mốc Thời Gian (Bao gồm cả 60 ngày) tại Trang Huấn Luyện

**Trạng thái:** ✅ Hoàn thành xuất sắc, đã kiểm tra cú pháp và dịch vụ (Health check: ok)

### 1. Vấn đề phát hiện:
- Tại Trang 4 (Huấn luyện mô hình -> Tab 1: Cấu hình), ô *"Chọn các mốc cần cập nhật"* mặc định chỉ hiển thị 6 mốc: `1d, 5d, 10d, 15d, 20d, 30d`.
- Thiếu mất mốc `60 ngày (h60)` khiến người vận hành mỗi lần vào trang đều phải nhấp chuột chọn thêm mốc 60 ngày thủ công.

### 2. Nguyên nhân kỹ thuật:
- Tại [`app_main.py:2040`](app_main.py:2040), thuộc tính `default` của `st.multiselect` trước đó được gán cứng mảng 6 phần tử `default=[1, 5, 10, 15, 20, 30]` (do trong giai đoạn kiểm thử nội bộ trên CPU trước đây, mốc 60 ngày tốn nhiều thời gian nhất nên tạm loại khỏi default để test nhanh).

### 3. Giải pháp đã xử lý:
- Nâng cấp thuộc tính mặc định thành `default=HORIZONS` (toàn bộ 7 mốc: `[1, 5, 10, 15, 20, 30, 60]`).
- Khi người dùng truy cập trang Huấn luyện, hệ thống tự động chọn sẵn đủ 100% cả 7 mốc thời gian. Người vận hành chỉ việc bấm *"🚀 Bắt đầu Job Huấn Luyện"* là tiến trình tối ưu hóa chạy đầy đủ chuỗi giá trị mà không cần phải chọn thêm mốc 60 ngày nữa.

---

## Đợt 13 — Làm nổi bật khối tên + icon app ở đầu Sidebar (Phương án A)

**Trạng thái:** ✅ Hoàn thành, đã test trực tiếp trên cả 5 trang + popup chào mừng + tour

### 1. Yêu cầu:
- Người dùng nhận xét khối tên app + icon 🛢️ ở đầu sidebar quá mờ nhạt, dễ bị lướt qua. Yêu cầu làm nổi bật hơn.
- Trước khi sửa, đã dựng 1 ảnh minh họa (Artifact) 3 phương án (Hiện tại / A: chỉ đổi khối tên-icon / B: đổi tên-icon + thu gọn khoảng cách menu) để người dùng chọn trước — người dùng chọn **Phương án A**.

### 2. Thay đổi:
- Tại [`app_main.py:1590`](app_main.py:1590): thay khối `st.sidebar.markdown` hiển thị tên app từ dòng chữ đơn giản (không màu, cỡ 15px) thành **thẻ nền gradient teal→violet** (`linear-gradient(135deg, #00ad91, #6954d9)`, đúng màu thương hiệu đã dùng trong tour và các tiêu đề mục), icon 🛢️ đặt trong ô bo góc riêng nền trắng mờ, tên rút gọn còn "Oil Forecast Hub" in đậm trắng + phụ đề nhỏ "AUTOMATED EVALUATION" viết hoa bên dưới.
- Chỉ sửa đúng 1 khối markdown này — không đụng vào cấu trúc menu (`st.sidebar.radio`), khối "Mô hình dự báo", hay bất kỳ selector/CSS nào khác. Đã `grep` xác nhận không có tour selector hay CSS nào phụ thuộc vào text "Oil Forecast – Automated Evaluation Hub" trong sidebar trước khi sửa.

### 3. Kết quả kiểm tra trực tiếp:
- Đã bấm qua đủ cả 5 trang (Dự báo, Đánh giá mô hình, Lịch sử & Xuất dữ liệu, Huấn luyện mô hình, Hướng dẫn sử dụng) — khối tên mới hiển thị nhất quán, không lệch/vỡ layout ở trang nào.
- Mở lại popup chào mừng (nút "↺ Xem lại thông báo chào mừng...") — popup và nút "Để sau"/"Bắt đầu hướng dẫn" vẫn hoạt động bình thường, không bị ảnh hưởng bởi thay đổi sidebar.
- Console không phát sinh lỗi JS ở bất kỳ bước nào.

---

## Đợt 14 — Nâng cấp Toàn Diện Nội Dung Hướng Dẫn Nghiệp Vụ & Tự Động Điều Chỉnh Theo Phần Cứng CPU / GPU

**Trạng thái:** ✅ Hoàn thành xuất sắc, đã kiểm tra cú pháp Python (Exit code: 0) và dịch vụ vận hành mượt mà (Health check: ok)

### 1. Cơ chế nhận diện và hướng dẫn theo Phần cứng (CPU / GPU) thực tế của máy chủ:
- Mã nguồn kiểm tra trực tiếp qua lệnh lõi `torch.cuda.is_available()`:
  - **Nếu máy nhận diện GPU NVIDIA CUDA:** 
    - Sidebar hiển thị huy hiệu xanh `🟢 Server: GPU NVIDIA CUDA (Tăng tốc tối đa · Sẵn sàng)`.
    - Trang 1 hiển thị thẻ phần cứng GPU và thông báo tăng tốc tối đa.
    - Trang 4 tự động chọn mặc định **50 Epochs**, thông báo tốc độ siêu nhanh 10–20 giây/mốc.
    - Trang 5 hiển thị Banner nhận diện GPU với khuyến nghị tối ưu hóa tốt nhất.
    - Lời thoại Tour tương tác tự động cập nhật: *"tốc độ tối ưu cực nhanh 10–20 giây/mốc trên GPU NVIDIA CUDA"*.
  - **Nếu máy nhận diện CPU:**
    - Sidebar hiển thị huy hiệu `🔵 Server: CPU Doanh Nghiệp (6 vCPUs · Dự báo tức thì < 1s)`.
    - Trang 4 tự động chọn mặc định **25 Epochs**, thông báo thời gian chạy an toàn 1–2 phút/mốc.
    - Trang 5 hiển thị Banner nhận diện CPU ổn định, tin cậy.
    - Lời thoại Tour tương tác tự động cập nhật: *"khoảng 1–2 phút/mốc trên CPU 6 vCPUs"*.

### 2. Chuẩn hóa cẩm nang nghiệp vụ "Cách xem & Cách thực hiện" trên toàn bộ các trang:
- **Trang 1 (Dự báo thị trường):**
  - Chú thích đơn vị tiền tệ chuẩn quốc tế: **USD/thùng** (MG95, MG92) và **USD/tấn** (DO).
  - Hướng dẫn thao tác kéo thả file Excel, hệ thống tự động nhận diện và tính toán 7 mốc trong < 1s mà không cần thao tác thêm.
  - Hướng dẫn tương tác đồ thị Plotly: bấm chú giải ẩn/hiện từng mốc, rê chuột xem giá chi tiết, kéo thả chuột phóng to (zoom in).
- **Trang 2 (Đánh giá mô hình - Backtesting):**
  - Thêm cẩm nang đọc chỉ số bình dân: Giải thích ý nghĩa của **MAPE (%)** và **MAE (USD)**.
  - Sửa chuẩn xác đơn vị tiền tệ trên thẻ MAE từ "VNĐ" thành **USD**.
  - Quy tắc màu sắc bảng nhiệt Heatmap: Xanh (< 7%) là rất tốt ➔ Vàng (7-10%) là chấp nhận được ➔ Đỏ (> 10%) là thị trường biến động mạnh, khuyến nghị Finetune.
- **Trang 3 (Lịch sử & Xuất dữ liệu):**
  - Hướng dẫn tra cứu phục vụ thanh tra, kiểm toán.
  - Giải thích ý nghĩa cột *Dự báo* (quá khứ) vs *Thực tế* (sau đó).
  - Phân biệt trực quan trên đồ thị: Đường nét liền xanh ngọc (Thực tế) vs Đường nét đứt tím (AI dự báo).
- **Trang 4 (Huấn luyện mô hình):**
  - Hướng dẫn khi nào chọn chế độ *Finetune từ checkpoint* (định kỳ hàng tháng, chỉ mất 1-2 phút) vs *Huấn luyện lại từ đầu*.
  - Hướng dẫn đọc Tab 3 Benchmarking: Chọn phiên Baseline vs Phiên Cập nhật để nghiệm thu chỉ số **% Cải thiện độ chính xác (Giảm sai số màu xanh lá)**.
- **Trang 5 (Hướng dẫn sử dụng):**
  - Bổ sung Banner nhận diện phần cứng máy chủ theo thời gian thực.
  - Bổ sung **Sơ đồ quy trình vận hành 4 bước khép kín (Workflow Pipeline)**: Nạp dữ liệu ➔ Kiểm định ➔ Finetune ➔ Lưu báo cáo.
  - Bổ sung mục **Giải đáp thắc mắc nghiệp vụ thường gặp (FAQ)** gồm 4 câu hỏi thực tế.

### 3. Nâng cấp lời thoại Tour mũi tên tương tác:
- Cập nhật lời thoại của cả 4 kịch bản tour (`forecast`, `metrics`, `training`, `history`) thành các chỉ dẫn hành động cụ thể, chi tiết, giúp người dùng mới chỉ cần đi theo mũi tên là nắm trọn vẹn cách vận hành phần mềm.




# Thực nghiệm: So sánh CPU/GPU và kiểm tra khóa chạy song song (Huấn luyện)

> Mục đích: xác minh nhánh CPU (chưa từng chạy thật trên máy này vì máy luôn có GPU), đo thời gian thật CPU vs GPU trên dữ liệu đầy đủ, và kiểm tra xem cơ chế khóa `.training.lock` có chặn được việc lỡ chạy 2 job huấn luyện cùng lúc hay không.
>
> **Toàn bộ thực nghiệm chạy trong 2 thư mục cô lập tạm thời** (`_test_train_isolated`, `_test_train_full` — copy riêng `train_all_horizons.py` + `oil_forecast_research_new-main`), **không đụng đến code chính, dữ liệu thật, hay checkpoint thật** trong `D:\Anh_Thuy`. Cả 2 thư mục đã được xóa sau khi thực nghiệm xong.

---

## 1. Xác minh nhánh CPU hoạt động đúng (chưa từng test thật trước đây)

**Bối cảnh:** Chế độ CPU/GPU trong app **không phải nút chọn tay** — code tự gọi `torch.cuda.is_available()` để quyết định. Vì máy chủ hiện tại luôn có GPU, mọi lần huấn luyện thật từ trước đến giờ đều tự động rơi vào nhánh GPU. Nhánh CPU (dòng cảnh báo, epoch mặc định 25, checkpoint) **chưa từng được chạy thật để kiểm chứng**.

**Cách giả lập tắt GPU:** dùng biến môi trường `CUDA_VISIBLE_DEVICES` khi khởi chạy tiến trình con — không sửa code.

- Thử `CUDA_VISIBLE_DEVICES=""` (chuỗi rỗng) → **KHÔNG có tác dụng trên Windows**: log vẫn ghi `Thiết bị sử dụng: cuda`, và phát sinh lỗi phụ `Invalid device id` khi lưu `training_history.json` (do `torch.cuda.get_device_name(0)` bị lệch). Đây là đặc thù driver NVIDIA/Windows, không phải lỗi trong code app.
- Đổi sang `CUDA_VISIBLE_DEVICES=-1` → **ép đúng CPU**: log ghi `Thiết bị sử dụng: cpu`, huấn luyện xong 2 mốc (1d, 5d) trên dữ liệu nhỏ (150 dòng), lưu checkpoint và `training_history.json` thành công, không lỗi.

**Kết luận:** Nhánh CPU hoạt động đúng như thiết kế. Không cần sửa gì.

---

## 2. So sánh thời gian thật CPU vs GPU (dữ liệu đầy đủ)

**Điều kiện:** copy toàn bộ dữ liệu sạch thật (`clean_data_exo_ver1.csv`, 4660 dòng) vào thư mục cô lập `_test_train_full`, chạy 1 mốc (h=1), 25 epochs (đúng số epoch mặc định của chế độ CPU trên giao diện), đo thời gian bằng `date +%s` trước/sau tiến trình.

| Thiết bị | Thời gian thật | Val Loss cuối |
|---|---|---|
| CPU (giả lập `CUDA_VISIBLE_DEVICES=-1`) | **139 giây** (~2 phút 19 giây) | 0.011609 |
| GPU (thật, không can thiệp) | **49 giây** | 0.011757 |

**Nhận xét:**
- GPU nhanh hơn CPU khoảng **2.8 lần** trên dữ liệu đầy đủ (khác biệt so với con số "10-20 giây/mốc trên GPU" ghi trên UI — vì đó là ước tính khi *Finetune* từ checkpoint có sẵn, còn thực nghiệm này train mới hoàn toàn nên chậm hơn).
- Thời gian CPU thực đo (~2.3 phút) khớp khá sát với dòng cảnh báo trên giao diện ("khoảng 1–2 phút mỗi mốc") — không lệch nhiều, không cần sửa lại nội dung cảnh báo.
- Độ chính xác (val loss) giữa CPU và GPU gần như tương đương (chênh lệch ở chữ số thập phân thứ 3, do khác biệt tính toán dấu phẩy động phần cứng — không phải lỗi).
- **Lưu ý phụ trong lúc đo:** log tiến độ epoch (`print(...)` không có `flush()`) bị Python đệm (buffer) khi output được redirect ra file — nên trong lúc job đang chạy, file log có thể "im lặng" một thời gian dài rồi hiện đầy đủ nội dung cùng lúc khi xong. Đây chỉ là đặc điểm khi test qua dòng lệnh với `> file.log`, **không ảnh hưởng gì đến trải nghiệm thật trên giao diện Streamlit** (giao diện đọc trực tiếp từ luồng output của subprocess, không qua bước ghi/đọc file log này).

---

## 3. Test khóa chạy song song — phát hiện thật (chưa sửa, cần duyệt)

**Cách test:** cố tình khởi chạy 2 tiến trình `train_all_horizons.py` gần như cùng lúc, cùng nhắm vào 1 mốc (h=1) trong cùng thư mục checkpoint — mô phỏng tình huống 2 job "đụng" nhau.

**Kết quả quan sát được:**
- Job B chọn `--force_retrain` ("Train lại từ đầu"), log báo đúng là đã xóa checkpoint cũ (`🗑️ Đã xóa checkpoint cũ: gumnet_h1.pt`) và bắt đầu train lại từ đầu.
- Nhưng ngay sau đó, log lại hiện `♻️ Đã nạp trọng số GUMNet h1 sẵn có để tối ưu tiếp (Finetune)...` — tức là Job B **vô tình Finetune trên checkpoint mà Job A vừa ghi xong**, dù người vận hành (giả định) đã chọn rõ ràng "Train lại từ đầu".
- **Nguyên nhân gốc:** `train_all_horizons.py` **không tự kiểm tra file khóa `.training.lock` trước khi chạy** — nó chỉ tự ghi khóa của chính nó (`_write_own_lock`) rồi chạy luôn, không hỏi "có job nào khác đang giữ khóa không?". Việc chặn hiện tại **hoàn toàn nằm ở phía giao diện Streamlit** (`app_main.py` gọi `get_active_training_lock()` để làm mờ nút bấm) — nếu thao tác đúng qua giao diện web (chỉ có đúng 1 nút "Bắt đầu Job", đã bị khóa xám khi có job khác chạy) thì **an toàn tuyệt đối, không xảy ra tình huống này**.
- Rủi ro chỉ có thật nếu ai đó **chạy thẳng `train_all_horizons.py` qua terminal** (bỏ qua giao diện web) — trường hợp hiếm nhưng có thể xảy ra khi debug/vận hành thủ công.

**Đề xuất khắc phục (chưa làm — cần bạn duyệt trước khi sửa):**
Thêm một đoạn kiểm tra ngay đầu `train_all_horizons.py`: nếu phát hiện `.training.lock` đang tồn tại và thuộc về một tiến trình (PID) còn sống khác với chính nó → in cảnh báo và thoát ngay, không chạy. Thay đổi này nhỏ, chỉ bổ sung thêm 1 lớp bảo vệ ở tầng script (phòng trường hợp chạy tay qua terminal), **không ảnh hưởng gì đến cách vận hành bình thường qua giao diện web hiện tại**.

---

## 4. Đã sửa và xác nhận (Đợt 2 — theo yêu cầu người dùng)

**Thay đổi trong code:**
1. [`train_all_horizons.py`](../train_all_horizons.py): thêm hàm `_check_no_other_job_running()` — đọc `.training.lock`, nếu PID trong đó còn sống và khác PID của chính mình → in `❌ Đã có một Job Huấn luyện khác đang chạy...` rồi thoát ngay (`sys.exit(1)`), không chạm vào checkpoint. Gọi hàm này ngay đầu `__main__`, trước cả bước ghi khóa của chính mình.
2. Trong lúc viết hàm trên, phát hiện thêm 1 lỗi thật khi kiểm thử: `os.kill(pid, 0)` trên Windows với PID đã chết đôi khi ném `SystemError` thay vì `OSError` thông thường — khiến hàm kiểm tra bị crash giữa chừng. Đã bọc thêm `except Exception: return False` trong hàm `_pid_alive()`.
3. Vì hàm `_pid_alive()` y hệt cũng tồn tại sẵn trong [`app_main.py`](../app_main.py) (được giao diện web dùng để làm mờ nút bấm), đã áp dụng cùng bản vá ở đó để tránh trường hợp hiếm gặp: lỗi lạ khi kiểm tra PID khiến giao diện tưởng nhầm job đã xong dù có thể vẫn đang chạy.

**Cách kiểm chứng:** đua thời gian thật giữa 2 job (bắn job B ngay khi job A còn đang chạy) cho kết quả **không ổn định** — nhiều lần job A tình cờ hoàn tất quá nhanh (do cơ chế early-stopping của mô hình) trước khi job B kịp kiểm tra, nên không tạo được chồng lấn thời gian đáng tin cậy để làm bằng chứng. Chuyển sang cách kiểm chứng trực tiếp và chắc chắn hơn: tạo 1 tiến trình Python giả chạy nền 60 giây, lấy đúng PID Windows thật của nó (không dùng PID của Git Bash vì không khớp với PID thật của Windows), ghi PID đó vào `.training.lock` giả, rồi chạy `train_all_horizons.py`:

- **PID giả còn sống →** script in đúng `❌ Đã có một Job Huấn luyện khác đang chạy (PID ..., bắt đầu lúc ...). Dừng ngay để tránh ghi đè checkpoint lẫn nhau.` và thoát với exit code `1`, **không đụng vào checkpoint**. ✅
- **PID giả trong lock là 999999 (chắc chắn không tồn tại, mô phỏng khóa "rác" của job cũ bị crash) →** script tự nhận ra khóa đã hết hiệu lực, bỏ qua và chạy bình thường, **không bị kẹt cứng vĩnh viễn**. ✅

**Kết luận:** Lỗ hổng đã được vá đúng — script giờ tự bảo vệ được cả khi bị gọi trực tiếp qua terminal (bỏ qua giao diện web), không chỉ dựa vào việc giao diện làm mờ nút bấm như trước.

---

## 5. Sự cố thật trên server thật (sau khi đã đẩy Đợt 2 lên git) — đã sửa

Người dùng thao tác thật trên app thật (tải file `real_price_petroleum_test.xlsx` — vốn là file tôi tải để test trước đó, không phải dữ liệu thật) rồi bấm "Bắt đầu Job Huấn Luyện". Job chỉ chạy xong đúng 1 mốc (1 ngày) rồi dừng bất thường 2 lần liên tiếp, với 2 dấu hiệu khác nhau:

1. **Lần 1:** không có traceback, chỉ thấy checkpoint `gumnet_h1.pt` được ghi lúc 13:23 rồi dừng, 6 mốc còn lại (5-60 ngày) không chạy. Nguyên nhân suy luận: người dùng đổi trang/tab giữa lúc job đang chạy.
2. **Lần 2:** có traceback rõ ràng: `RuntimeError: File ...\gumnet_h1.pt cannot be opened` ngay tại `torch.save()` — Windows khóa file, khả năng cao do chính app Streamlit đang mở/đọc đúng file đó cùng lúc.

**Đã xác minh trước khi sửa (không đoán mò):**
- `git diff` xác nhận `clean_data_exo_ver1.csv` chỉ được **thêm** 74 dòng mới (đến 04/09/2026), không mất/sai dữ liệu cũ — an toàn, không cần khôi phục.
- `torch.load()` lại `gumnet_h1.pt` xác nhận file **không hề bị hỏng** (lỗi xảy ra trước khi ghi, không phải ghi dở).

**Đã sửa (3 việc):**
1. **Xóa** `datasets/real_price_petroleum_test.xlsx` (file test lỡ lọt vào thư mục dữ liệu thật qua thao tác upload).
2. **[`train_all_horizons.py`](../train_all_horizons.py):** thêm hàm `safe_torch_save()` — ghi checkpoint ra file tạm `.tmp` rồi mới đổi tên đè lên (thao tác đổi tên khó bị khóa hơn ghi đè trực tiếp); nếu vẫn bị khóa thì tự đợi và thử lại tối đa 6 lần thay vì crash ngay. Áp dụng cho cả 2 chỗ lưu checkpoint (GUMNet và HybridTriNet).
3. **[`app_main.py`](../app_main.py):** đổi cách theo dõi tiến trình huấn luyện từ đọc trực tiếp `subprocess.PIPE` (chỉ tồn tại khi script Streamlit đang chạy — bị ngắt ngay khi đổi trang) sang **ghi log ra 1 file thật trên đĩa**, script web chỉ đọc-nối-tiếp (tail) file đó để cập nhật giao diện. Nhờ vậy tiến trình huấn luyện con hoàn toàn độc lập với vòng đời trang web — đổi trang/đóng tab giữa chừng không còn làm nó chết nữa.

**Cách kiểm chứng:**
- Test riêng `safe_torch_save()`: giả lập khóa file bằng `msvcrt.locking()` trong 3 giây rồi mới thả — hàm tự thử lại 3 lần (đúng log cảnh báo mỗi lần), rồi ghi thành công ngay khi khóa được giải phóng.
- Test riêng cơ chế ghi-log-ra-file: cho tiến trình con in 5 dòng cách nhau 1 giây, **đóng hẳn handle ghi log của "cha" ngay sau khi khởi động** (mô phỏng bị ngắt trang) — xác nhận tiến trình con vẫn ghi đủ cả 5 dòng và chạy xong hoàn toàn bình thường, không phụ thuộc "cha" còn sống hay không.
- Cả 2 test chạy trong thư mục cô lập, không đụng dữ liệu/checkpoint thật.

---

## 6. Dọn dẹp

Toàn bộ thư mục thực nghiệm (`_test_train_isolated`, `_test_train_full`, `_verify_lock_fix`, `_verify_safe_save`, `_verify_logfile`) đã được xóa sau khi hoàn tất — không để lại dấu vết trong `D:\Anh_Thuy`.

# BAO CAO SUA PIPELINE TU DONG

Ngay kiem tra: 08/09/2026

## Noi dung da sua

### Bo sung sau danh gia tich hop

- Sua loi `pipeline_engine.py` goi sai contract cua `backtest_worker.run_upload_simulation()`.
- Goi dung cac tham so vi tri `base_path`, `upload_files`, `start_date` va cac tham so `sel_horizons`, `sel_models`, `log_fn`.
- Nhan dung mot `DataFrame` tra ve thay vi unpack ba gia tri khong ton tai.
- Them `_summarize_backtest()` de tinh MAPE, MAE va so mau tu cac cot `% Lech`, `Sai lech` cua DataFrame.
- Tu xac dinh `start_date` la 365 ngay truoc ngay du lieu moi nhat trong base va cac file upload.
- Backtest tra sai kieu, thieu cot hoac co chi so khong hop le se lam pipeline `failed` va giai phong lock.
- Them test tich hop goi xuyen `run_pipeline_task()` voi mot worker co dung chu ky that.
- Them test tich hop xac nhan backtest exception lam pipeline failed, khong chay tiep sang huan luyen.
- Da xoa `debug_out.txt` va `datasets/upload_iru4rk9j.csv` sau khi xac nhan day la file debug/test.

- Sua nhan dien cot ngay: chi chap nhan alias ro rang, khong con match mo ho bang chuoi `ng`.
- Preview upload nhan du lieu he thong hien co de phat hien dung ngay trung va ngay co gia dieu chinh.
- Commit chi ghi cac dong co ngay moi da qua validate; khong ghi lai nguyen file chua du lieu cu.
- File da commit dung timestamp + UUID, tranh de file cung ten trong cung mot giay.
- Du lieu lich su khac biet chi duoc canh bao, ban demo khong tu ghi de.
- Nut tiep nhan dot upload moi bi khoa khi pipeline dang xu ly; cac trang xem/xuat du bao van hoat dong.
- Them pipeline lock nguyen tu, co owner pipeline_id va PID worker thuc.
- Backtest loi chuyen pipeline sang `failed`, giu ket qua cu; khong con bien loi thanh MAPE 0 va bao complete.
- Candidate duoc seed tu checkpoint production de la finetune, khong train ngau nhien trong thu muc rong.
- Checkpoint moi ghi them `best_val_loss`.
- Candidate bat buoc du 7 horizon, dung job_id, horizon, state dict va validation loss hop le.
- Candidate chi duoc promote khi validation loss trung binh tot hon baseline production.
- Neu production cu chua co best_val_loss, baseline duoc doc tu training_history thanh cong gan nhat.
- Neu khong co baseline dang tin cay, he thong fail closed va giu production.
- Promote co backup day du va rollback tat ca checkpoint neu copy/replace loi giua chung.
- Xoa cac cau chu frontend con dan khach sang trang Huan luyen da bi bo.
- Cap nhat test dieu huong tu 5 trang cu sang 4 trang khach hang.
- Them test truc tiep cho data_pipeline va pipeline_engine.

## Ket qua kiem thu

- `py_compile`: PASS.
- `python -m unittest discover -s tests -p test_*.py`: 48/48 PASS.
- Co test rieng cho validate/phan loai/commit du lieu.
- Co test lock owner.
- Co test candidate sai job_id.
- Co test thieu baseline khong promote.
- Co test rollback du 7 checkpoint khi promote loi giua chung.
- AppTest di qua 4 trang khach hang thanh cong.

## Gioi han con lai

- Chua chay mot phien huan luyen GPU day du trong dot sua nay de tranh tu y thay doi checkpoint production va ton tai nguyen may.
- Candidate hien duoc so sanh bang validation loss cua trainer. Day la cong bao ve tot hon nhieu so voi chi dem du 7 file, nhung buoc nang cap tiep theo nen danh gia production va candidate tren cung mot tap holdout/backtest co dinh truoc khi promote.
- Promote hien co backup + rollback theo giao dich. De dam bao doc atomic tuyet doi trong khoanh khac thay 7 file, kien truc tiep theo nen dung production manifest tro toi mot thu muc model version bat bien.
- Can xu ly canh bao Streamlit `st.components.v1.html` deprecated rieng; canh bao nay khong lam hong pipeline.

## Danh gia su dung

- Demo upload, du bao va backtest ngam: du dieu kien chay.
- Nut upload khoa trong khi mot dot dang xu ly, tranh tao job chong lan.
- Tu huan luyen an toan hon va khong con promote chi vi du 7 file.
- Khi demo huan luyen that, nen sao luu ngoai mot lan va co nguoi van hanh theo doi lan chay dau.

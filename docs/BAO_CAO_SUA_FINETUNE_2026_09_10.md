# Sua luong Finetune va danh gia

## Hanh vi moi

1. Backtest Production truoc huan luyen de lay baseline MAPE/MAE.
2. Tao ban sao CSV rieng trong thu muc candidate; gop du lieu datasets vao ban sao bang --data-path va --update_data.
3. Huan luyen candidate vao thu muc rieng va kiem tra du checkpoint/job_id.
4. Backtest candidate tu chinh thu muc candidate. So sanh dung cung Model, Horizon, Upload, Ngay, Target va gia thuc te, ke ca so luong diem.
5. Chi ap dung khi MAPE candidate thap hon. Truong hop bang nhau, kem hon, thieu diem hoac loi deu giu Production.
6. Sau ap dung, nap lai checkpoint va backtest kiem tra. Neu ket qua khong khop hoac loi, khoi phuc checkpoint va cache truoc huan luyen.
7. Luu dau du lieu da huan luyen trong .last_training.json sau khi huan luyen va danh gia thanh cong, ke ca candidate bi tu choi. Dong thong bao khong xoa dau nay. Du lieu thay doi thi cho phep toi uu tiep; luot loi van co the thu lai.

## Cac thay doi ho tro

- Cache model worker phan biet thu muc checkpoint va phien ban file; cache Streamlit cung nhan phien ban checkpoint.
- Fingerprint datasets dung SHA-256 noi dung thay vi chi dem file va lay thoi gian sua lon nhat.
- Helper backtest cu kiem tra PID cua lock va su ton tai/cache hop le truoc khi tin trang thai success. Helper nay van khong duoc kich hoat tu dong.
- Kiem tra PID Windows dung OpenProcess voi quyen SYNCHRONIZE va WaitForSingleObject timeout 0; khong dung os.kill tren Windows.
- Khong con dung validation loss lich su lam quyet dinh ap dung; baseline la ket qua Production vua do tren cung diem doi chieu.

## Kiem thu va gioi han

Ket qua: 70 passed trong 76.49 giay. Lenh: python -m pytest tests/test_candidate_flow.py tests/test_pipeline_engine.py tests/test_project_io.py tests/test_data_pipeline.py tests/test_upload_flow.py tests/test_backtest_job.py tests/test_backup_restore.py tests/test_lock_concurrency.py -q -p no:cacheprovider. Day la bo test lien quan da chon, khong phai toan bo tests/. py_compile 6 module thay doi va git diff --check deu thanh cong.

tests/test_candidate_flow.py kiem thu run_pipeline_task trong thu muc tam voi trainer/backtest gia lap: candidate tot hon, kem hon, bang nhau, loi danh gia, thieu diem, loi kiem tra sau ap dung, loi trainer; kiem tra checkpoint/cache/lock va ngan lap sau khi dong thong bao.

Co test gop gia upload vao CSV candidate, giu CSV goc; noi dung doi nhung timestamp giu nguyen; cache checkpoint doi phien ban; kiem tra PID nhieu lan khong lam dung tien trinh con that.

Chua chay 7 moc x 30 epoch tren du lieu that. Cac test mo phong luong khong chung minh thoi gian, bo nho hay chat luong hoc that. Backtest tren lich su da dung de huan luyen la phep doi chieu hoi cuu, khong phai bang chung cai thien tren du lieu tuong lai chua tung hoc. Muon do kha nang tong quat hoa can tap holdout tach rieng theo thoi gian.

Khong dat lai datasets, checkpoint Production hay chi so theo con so mong muon. Cache cu dung fingerprint cu se duoc danh dau can cap nhat; chi so moi phai do backtest tinh ra.

Can nap lai server bang code moi truoc khi test thu cong, vao thoi diem khong con job dang chay. Phien lam viec nay khong tu restart server va khong push Git.

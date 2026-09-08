# KE HOACH FRONTEND VA LUONG XU LY TU DONG

## 1. Muc tieu

Tai lieu nay la dac ta trien khai cho ban demo production chay trong mang LAN. Ung dung chi su dung GUMNet tren giao dien khach hang. Khach hang khong can biet ky thuat, khong chon model, khong chon checkpoint va khong tu cau hinh huan luyen.

Trai nghiem muc tieu:

1. Khach chon mot hoac nhieu file Excel/CSV.
2. Khach bam mot nut `Kiem tra & cap nhat`.
3. He thong kiem tra tung file, chi luu du lieu hop le.
4. He thong cap nhat du bao ngay bang GUMNet production hien tai.
5. Backtest tu dong chay ngam de doi chieu du bao cu voi gia thuc te moi.
6. Chi khi du du lieu va chat luong model giam qua nguong, he thong moi huan luyen GUMNet candidate.
7. Candidate chi thay production neu tot hon va vuot tat ca kiem tra.
8. Neu bat ky buoc nao loi, production hien tai van phai tiep tuc phuc vu du bao.

Khong duoc trien khai theo kieu upload xong roi ghi de truc tiep checkpoint production.

## 2. Pham vi ban demo

### Co trong ban demo

- Upload mot hoac nhieu file `.xlsx`, `.xls`, `.csv`.
- Kiem tra schema va chat luong du lieu truoc khi luu.
- Phan biet du lieu cu, moi, trung lap va du lieu dieu chinh.
- Du bao bang GUMNet production hien tai.
- Backtest tu dong chay ngam.
- Tu quyet dinh co can huan luyen lai hay khong.
- Huan luyen GUMNet candidate chay ngam.
- So sanh candidate voi production.
- Promote candidate hoac giu production cu.
- Lich su cua moi dot xu ly.
- Xuat ket qua CSV/Excel.
- Thong bao loi than thien, khong hien traceback mac dinh.

### Khong co tren giao dien khach

- Chon HybridTriNet.
- Chon checkpoint.
- Chon model production.
- Chon epoch, learning rate hoac tham so ky thuat.
- Nut huan luyen thu cong.
- Nut khoi phuc checkpoint.
- Log Python va stack trace mo san.

Hybrid co the con trong source de nghien cuu, nhung tat ca entry point cua ung dung production phai dung allowlist `GUMNet`. Khong duoc tu dong fallback sang Hybrid khi GUMNet loi.

## 3. Cau truc frontend

Menu khach hang chi gom bon trang:

1. `Du bao`
2. `Danh gia mo hinh`
3. `Lich su & Xuat du lieu`
4. `Huong dan su dung`

Bo trang `Huan luyen mo hinh` khoi menu khach. Cong cu ky thuat neu can phai nam trong script/CLI rieng, khong nam trong luong demo.

### Thanh trang thai job toan cuc

Khi co dot xu ly dang chay, hien mot thanh trang thai tren moi trang:

```text
Cap nhat du lieu thang 09/2026

[x] Da kiem tra 3 file
[x] Da ghi nhan 12 ngay moi
[>] Dang doi chieu du bao voi thuc te
[ ] Toi uu GUMNet neu can
[ ] Hoan tat

Ban van co the xem va xuat du bao trong luc he thong xu ly.
```

Thanh trang thai phai doc trang thai tu kho luu ben vung, khong chi dua vao `st.session_state`. Chuyen trang, rerun hoac mo mot trinh duyet LAN khac khong duoc lam mat job.

## 4. Trang Du bao

### Noi dung khach thay

- Ten san pham va trang thai `GUMNet dang san sang`.
- Ngay du lieu moi nhat trong he thong.
- Vung keo tha/chon nhieu file.
- Danh sach xem truoc tung file.
- Nut chinh `Kiem tra & cap nhat`.
- Tien trinh dot upload.
- Bang va bieu do du bao cho cac moc 1, 5, 10, 15, 20, 30, 60 ngay.
- Nut xuat CSV/Excel.

### Xem truoc tung file

Truoc khi bam xu ly, moi file hien:

- Ten file.
- Dinh dang.
- So dong doc duoc.
- Ngay nho nhat va lon nhat neu doc duoc.
- Cac cot gia tim thay.
- Trang thai so bo: san sang hoac can sua.

Khong luu file vao `datasets/` trong buoc preview.

### Trang thai nut

- Chua chon file: nut bi khoa.
- Co file: `Kiem tra & cap nhat N file`.
- Dang kiem tra: `Dang kiem tra...`, khong bam lai duoc.
- Tat ca file sai: mo khoa sau khi hien loi.
- Da chap nhan dot upload: `Da tiep nhan`, khong bam lai cung noi dung file.
- Neu nguoi dung thay doi/them file: mo khoa cho dot moi.

Nhan dien file bang SHA-256 noi dung, khong chi bang ten hoac kich thuoc.

## 5. Kiem tra file

Moi file phai duoc kiem tra doc lap. Mot file sai khong duoc lam huy cac file dung.

### Kiem tra bat buoc

- Phan mo rong nam trong allowlist.
- Ten file duoc lam sach, khong cho path traversal.
- Kich thuoc khong vuot gioi han cau hinh.
- File mo va parse duoc.
- File khong rong.
- Co cot `Ngay` sau khi chuan hoa ten cot.
- Cot ngay chuyen duoc sang datetime.
- Co it nhat mot cot gia muc tieu duoc ho tro.
- Cot gia co du lieu so hop le.
- Khong chua gia vo cuc.
- Quy tac gia am, bang 0 hoac bien dong bat thuong phai duoc canh bao/tu choi theo cau hinh nghiep vu.
- Khong co cong thuc, macro hoac noi dung khong can thiet duoc thuc thi.

### Ket qua tung file

```text
tuan_1.xlsx   Hop le - 5 ngay moi
tuan_2.xlsx   Hop le - 5 ngay moi
sai_form.xlsx Khong hop le - thieu cot Ngay
```

Chi file hop le duoc dua sang buoc phan loai du lieu.

## 6. Phan loai du lieu

So sanh du lieu trong file voi kho du lieu chinh theo ngay va mat hang.

### Truong hop A: chi co du lieu moi

Vi du he thong co den `15/09/2026`, file co `16/09/2026-20/09/2026`.

- Chap nhan va hop nhat.
- Tao mot job cho ca dot, khong tao mot job cho moi file.
- Cap nhat ngay du lieu moi nhat.
- Chay du bao ngay bang production hien tai.
- Dua backtest vao hang doi.

### Truong hop B: chi co du lieu cu/trung khop

- Khong luu ban sao trung lap.
- Khong backtest lai.
- Khong huan luyen.
- Hien `File hop le nhung khong co du lieu moi`.
- Ket thuc va mo khoa nhanh.

### Truong hop C: vua co ngay cu vua co ngay moi

- So sanh tung gia tri cua phan ngay cu.
- Gia cu giong he thong: bo qua nhu ban ghi trung.
- Gia cu khac he thong: danh dau la du lieu dieu chinh.
- Ngay moi: chap nhan neu hop le.
- Khong am tham ghi de lich su.

Trong ban demo, neu co du lieu dieu chinh, frontend phai hien tom tat va yeu cau mot lan xac nhan nghiep vu truoc khi cap nhat. Neu khong co dieu chinh thi xu ly tu dong.

### Truong hop D: nhieu file chong lan ngay

- Gop file theo thu tu xac dinh, khong phu thuoc thu tu filesystem.
- Neu hai file cung mot ngay/mat hang va cung gia: gop mot ban ghi.
- Neu khac gia: bao xung dot, khong tu chon ngau nhien.
- Cac file/ban ghi khong xung dot van duoc giu de xu ly neu giao dich du lieu cho phep tach rieng.

### Truong hop E: tat ca file sai

- Khong ghi gi vao `datasets/`.
- Khong tao backtest/training job.
- Khong thay doi fingerprint/cache.
- Hien loi rieng tung file va nut chon lai.

### Truong hop F: mot so file dung, mot so file sai

- Xu ly file dung.
- Giu thong bao file sai sau rerun.
- Khong hien `tat ca da xu ly xong` neu file sai van dang nam trong dot chon.
- Hien ro `Da cap nhat X file; Y file khong duoc chap nhan`.

## 7. Giao dich cap nhat du lieu

Du lieu hop le phai duoc ghi an toan:

1. Chuan hoa trong thu muc staging.
2. Kiem tra lai schema sau chuan hoa.
3. Tinh fingerprint cua bo du lieu du kien.
4. Ghi file tam.
5. Dung thao tac atomic replace khi co the.
6. Chi cap nhat manifest/fingerprint sau khi tat ca file can thiet da ghi thanh cong.
7. Neu loi, rollback staging va giu kho du lieu cu.

Moi dot tao `ingestion_job_id`. Trang thai toi thieu:

```json
{
  "job_id": "...",
  "fingerprint": "...",
  "status": "validating|waiting_confirmation|committing|queued|failed|complete",
  "files": [],
  "new_rows": 0,
  "corrected_rows": 0,
  "created_at": "...",
  "updated_at": "...",
  "error_code": null
}
```

## 8. Du bao sau upload

Ngay khi du lieu da commit:

- Du bao bang GUMNet production hien tai.
- Khong cho candidate dang huan luyen phuc vu request.
- Cache du bao gan voi `data_fingerprint` va `production_model_version`.
- Khi du lieu/model thay doi, cache lien quan phai duoc vo hieu hoa co kiem soat.

Khach khong can doi huan luyen xong de xem du bao.

## 9. Backtest chay ngam

### Muc dich

Dung gia thuc te moi de doi chieu voi cac du bao da tao truoc do co ngay dich da den han.

### Chi so can tinh

- MAE tong.
- MAPE tong.
- MAE/MAPE theo mat hang.
- MAE/MAPE theo horizon.
- So cap du bao-thuc te hop le.
- Khoang ngay danh gia.
- Phien ban production tao du bao cu.

### Quy tac

- Backtest bat dau sau khi commit du lieu thanh cong.
- Chay bang subprocess/worker doc lap, khong bang Thread chi song theo Streamlit session.
- Mot fingerprint chi co toi da mot job dang chay.
- Ghi trang thai ra JSON/SQLite bang thao tac atomic.
- Ghi cache qua file tam va `os.replace`.
- Neu co fingerprint moi trong luc dang chay, xep mot job moi nhat tiep theo; khong tao vo han job trung.
- Neu loi, giu ket qua backtest cu va danh dau stale.

### Trang Danh gia khi job chay

- `pending/running`: hien ket qua cu kem dong `Dang tu dong cap nhat`.
- `success` dung fingerprint: hien ket qua moi, bo canh bao stale.
- `failed`: giu ket qua cu, hien thong bao than thien va nut `Thu lai`.
- Nut cap nhat thu cong chi xuat hien khi failed/stale, khong phai thao tac bat buoc.

## 10. Dieu kien huan luyen lai

Khong huan luyen chi vi co file moi. Bo quyet dinh chi tao training job khi tat ca dieu kien dat:

- Co ban ghi moi hop le.
- Co du so diem moi toi thieu theo cau hinh.
- Co du cap du bao-thuc te toi thieu de danh gia dang tin cay.
- Backtest cua fingerprint moi da thanh cong.
- MAPE hoac quy tac chat luong vuot nguong can toi uu.
- Khong co training job dang chay.
- Backup production thanh cong va da duoc xac thuc.
- Tai nguyen he thong dat nguong an toan.

Cac gia tri nhu `minimum_new_rows`, `minimum_evaluation_pairs`, `mape_retrain_threshold` phai nam trong mot file cau hinh tap trung. Can chot voi chu san pham truoc khi gan gia tri chinh thuc.

Neu khong du dieu kien, ghi ro ly do:

- `Khong can toi uu: MAPE van trong nguong tot`.
- `Chua du du lieu moi de danh gia dang tin cay`.
- `Dang cho job hien tai hoan tat`.

## 11. Huan luyen GUMNet candidate

### Nguyen tac

- Production la read-only trong suot training.
- Tao thu muc candidate rieng theo `training_job_id`.
- Tat ca horizon duoc luu vao candidate, khong ghi tung horizon vao production.
- Moi checkpoint chua `job_id`, `horizon`, feature schema, scaler, model state va version.
- Neu mot horizon loi/skip/thieu, ca candidate that bai.

### Cac buoc

1. Khoa training job theo atomic lock.
2. Xac thuc PID/job hien tai neu tiep quan lock cu.
3. Sao luu du production day du.
4. Xac thuc backup doc/nạp duoc; backup loi thi khong duoc huan luyen.
5. Tao candidate workspace.
6. Huan luyen lan luot cac horizon da cau hinh.
7. Kiem tra tat ca checkpoint candidate.
8. Chay danh gia candidate tren tap kiem dinh co dinh.
9. So sanh cung tap du lieu voi production.
10. Promote hoac loai candidate.
11. Ghi lich su va giai phong lock trong `finally`.

### Tien trinh frontend

```text
Dang toi uu GUMNet

[x] Da tao ban an toan
[x] Moc 1 ngay
[x] Moc 5 ngay
[>] Moc 10 ngay
[ ] Moc 15 ngay
[ ] Moc 20 ngay
[ ] Moc 30 ngay
[ ] Moc 60 ngay
```

Khong hien epoch/log chi tiet cho khach. Khach van xem du bao bang production cu.

## 12. Danh gia va promote candidate

Candidate khong duoc promote chi vi script return code 0.

Bat buoc kiem tra:

- Du checkpoint cho moi horizon.
- Moi checkpoint thuoc dung `job_id`.
- Horizon trong checkpoint trung voi ten file.
- Schema feature/target hop le.
- State dict khong rong va nap duoc bang loader an toan.
- File duoc tao sau thoi diem job bat dau.
- Khong co NaN/Inf trong output kiem tra.
- Candidate co chi so tot hon production theo quy tac da chot.
- Khong lam xau di bat thuong mot mat hang/horizon quan trong.

Promote phai theo manifest/version:

1. Dat candidate hoan chinh trong thu muc version moi.
2. Xac thuc lai version moi.
3. Atomic switch con tro/manifest production sang version moi.
4. Neu switch loi, con tro cu van con nguyen.

Khong copy tung checkpoint truc tiep de tao mot bo production nua cu nua moi.

### Ket qua frontend

- Tot hon: `Da cap nhat phien ban GUMNet moi`.
- Khong tot hon: `Da kiem tra; GUMNet hien tai van cho ket qua tot hon`.
- That bai: `Khong the hoan tat toi uu; du bao hien tai van hoat dong`.

## 13. Khoa va hang doi

### Khoa upload

Khoa nut cua dot file hien tai tu khi bam den khi:

- Tat ca file bi tu choi; hoac
- Du lieu da commit va background job da duoc tao ben vung.

Khong can khoa nut den khi training ket thuc.

### Khoa backtest

- Chi mot writer vao cache.
- Khong tao job trung fingerprint.
- Fingerprint moi duoc coalesce thanh job moi nhat dang cho.

### Khoa training

- Chi mot training job toan he thong.
- Khong duoc xoa lock cua job khac.
- Chi takeover khi PID khong con song hoac heartbeat het han va thoa quy tac timeout.

### Upload moi khi job dang chay

Cho phep chon va kiem tra file. Sau khi commit:

- Neu backtest dang chay fingerprint cu, danh dau fingerprint moi cho luot ke tiep.
- Neu training candidate cu dang chay, khong chen du lieu moi vao candidate do.
- Candidate cu duoc danh gia theo snapshot luc bat dau.
- Sau khi xong, he thong xem lai fingerprint moi va quyet dinh co can chu ky tiep theo.

## 14. Trang Danh gia mo hinh

### Khach thay

- MAPE va MAE tong.
- Chi so theo mat hang.
- Chi so theo horizon.
- Bieu do du bao so voi thuc te.
- So mau danh gia.
- Khoang ngay danh gia.
- Thoi gian cap nhat gan nhat.
- Phien ban GUMNet production.
- Trang thai dang cap nhat/toi uu.

### Cach dien giai

Khong tuyen bo `chinh xac cao` chi dua vao mau giao dien. Mau xanh/vang/do phai dua tren nguong cau hinh da thong nhat.

- Xanh: chat luong trong nguong chap nhan.
- Vang: can theo doi them.
- Do: chat luong vuot nguong, he thong dang/da xem xet toi uu.

Neu so mau qua it, phai hien `Chua du du lieu de ket luan`, khong hien danh gia xanh gia tao.

## 15. Trang Lich su & Xuat du lieu

Moi dot hien:

- Job ID rut gon.
- Ten va hash file.
- Thoi diem tiep nhan.
- Khoang ngay.
- So dong moi/trung/dieu chinh/bi loai.
- Trang thai ingestion.
- Trang thai backtest va cac chi so chinh.
- Ly do co/khong huan luyen.
- Trang thai training.
- Candidate duoc promote hay production duoc giu.
- Model version truoc va sau.
- Nut xuat bao cao.

Khong hien nut xoa, sua checkpoint hoac restore cho khach demo.

## 16. Trang Huong dan

Noi dung gom:

- Tai file mau.
- Cot `Ngay` va cac cot gia duoc ho tro.
- Vi du file hop le/khong hop le.
- Cach doc du bao 1-60 ngay.
- Cach doc MAE/MAPE va so mau.
- Giai thich he thong tu doi chieu va chi toi uu khi can.
- Giai thich du bao van dung duoc khi he thong dang toi uu.

Bo toan bo huong dan chon model, epoch, Finetune thu cong va Hybrid.

## 17. Loi va thong bao

Frontend chi hien thong bao nghiep vu. Log ky thuat ghi vao file gan voi job ID.

Moi loi can co:

- Ma loi on dinh.
- Thong bao than thien.
- Anh huong den du lieu/model.
- Hanh dong tiep theo.
- Job ID de ky thuat tra cuu.

Vi du:

```text
Khong the doc file gia_thang_9.xlsx.
Du lieu va mo hinh hien tai khong bi thay doi.
Hay dung file mau va thu lai. Ma tham chieu: UPLOAD-AB12.
```

Khong dua traceback, duong dan noi bo hoac noi dung checkpoint ra giao dien mac dinh.

## 18. Trang thai job

Nen dung state machine ro rang:

### Ingestion

`created -> validating -> waiting_confirmation -> committing -> queued -> complete`

Nhanh loi: `validating/committing -> failed`.

### Backtest

`pending -> running -> success|failed|superseded`.

### Training

`pending -> backing_up -> training -> validating -> comparing -> promoting -> success`.

Nhanh khong can train: `pending -> skipped` kem reason.

Nhanh loi: bat ky state nao -> `failed`; production version khong doi.

Khong dung mot boolean nhu `is_running` lam nguon su that duy nhat.

## 19. Luu tru trang thai

Cho ban LAN mot may, co the dung SQLite de tranh race khi nhieu session Streamlit cung truy cap. Neu tam dung JSON:

- Moi file trang thai phai ghi qua temp + `os.replace`.
- Co lock doc/ghi ro rang.
- Khong dung pickle cho du lieu co the bi thay doi tu ben ngoai.
- Tat ca job co timestamp, fingerprint va version.
- Startup phai reconcile job `running` voi PID/heartbeat thuc te.

## 20. Khoi dong lai app/server

Sau restart:

- App doc kho trang thai ben vung.
- Neu worker con song, tiep tuc hien tien trinh.
- Neu worker chet, job chuyen failed/recoverable sau khi het heartbeat.
- Khong tu dong coi job thanh cong vi thay checkpoint cu.
- Khong xoa lock cua tien trinh dang song.
- Production manifest cu van phuc vu du bao.

## 21. Kiem thu bat buoc

### Upload

- Mot file hop le co ngay moi.
- Nhieu file deu hop le.
- Mot dung mot sai.
- Tat ca sai.
- File rong.
- Thieu cot Ngay.
- Ngay khong parse duoc.
- Thieu cot gia.
- Cung ten/cung kich thuoc nhung noi dung khac.
- File chi co du lieu cu.
- File vua cu vua moi.
- Hai file trung du lieu.
- Hai file xung dot gia cung ngay.
- Upload lai cung noi dung sau rerun.

### Backtest

- Tu dong tao job sau commit.
- Chuyen trang ngay khi job chay.
- Rerun khong mat job.
- Hai session khong tao job trung.
- Cache cu duoc giu khi job loi.
- Du lieu doi trong luc job chay.
- Restart Streamlit giua job.

### Training

- Khong huan luyen khi MAPE tot.
- Khong huan luyen khi thieu mau.
- Backup loi thi job khong bat dau.
- Hai process tranh lock, chi mot job thang.
- Mot horizon loi thi candidate that bai.
- Checkpoint cu khong duoc tinh la ket qua job moi.
- Sai `job_id`, horizon hoac schema bi tu choi.
- Candidate kem hon thi production khong doi.
- Candidate tot hon thi promote mot lan atomic.
- Loi khi promote thi rollback/con tro cu con nguyen.
- Restart app va restart worker giua training.

### Frontend

- Bon trang tai thanh cong tren desktop.
- Khong co trang/nut Hybrid tren UI khach.
- Khong co nut huan luyen thu cong.
- Thanh trang thai dung qua chuyen trang.
- Khong khoa xem/xuat du bao khi worker chay.
- Thong bao tung file khong mat sau rerun.
- Text/nut khong tran hoac chong len nhau.
- Console khong co loi app; warning thu vien phai duoc ghi ro neu chua xu ly duoc.

## 22. Kich ban demo nghiem thu

1. Khoi dong app moi.
2. Mo trang Du bao va xac nhan GUMNet production san sang.
3. Chon hai file hop le va mot file thieu cot Ngay.
4. Xac nhan preview dung va file sai duoc danh dau rieng.
5. Bam `Kiem tra & cap nhat` mot lan.
6. Xac nhan file sai khong vao `datasets/`, file dung duoc commit.
7. Xac nhan du bao moi xuat hien bang production hien tai.
8. Chuyen ngay sang trang Danh gia, khong bam nut thu cong.
9. Xac nhan thanh trang thai backtest van tiep tuc.
10. Cho job xong va xac nhan chi so tu cap nhat.
11. Neu du dieu kien training, xac nhan candidate chay ngam va du bao van dung duoc.
12. Xac nhan ket qua cuoi ghi `giu production` hoac `cap nhat GUMNet` ro rang.
13. Mo Lich su va doi chieu day du thong tin dot vua chay.
14. Upload lai file cu va xac nhan khong tao job thua.
15. Kiem tra console va thu muc test/rac.

## 23. Dieu kien chap nhan

Chi coi la hoan thanh khi:

- Khach chi can chon file va bam mot nut.
- File sai khong lam sap app va khong lam mat file dung.
- Du lieu cu khong kich hoat xu ly/huan luyen thua.
- Backtest chay ngam va ben vung qua rerun/chuyen trang.
- Huan luyen chi chay theo dieu kien, khong chay voi moi upload.
- Training khong ghi truc tiep vao production.
- Candidate loi/kem hon khong thay doi production.
- Promote la atomic theo mot version day du.
- Moi job co lich su va ly do quyet dinh.
- Giao dien khach chi co GUMNet va bon trang da quy dinh.
- Tat ca test tren pass va kich ban demo that dat.

## 24. Thu tu trien khai de han che rui ro

1. Dong bang hanh vi hien tai va bo sung test hoi quy.
2. Tach logic validate/phan loai/hop nhat file khoi Streamlit UI.
3. Tao kho job + state machine ben vung.
4. Chuyen backtest sang worker/subprocess.
5. Sua frontend thanh bon trang va thanh trang thai toan cuc.
6. Tao candidate workspace va production manifest.
7. Chuyen trainer sang chi ghi candidate.
8. Them compare/promote/rollback atomic.
9. Noi bo quyet dinh tu dong huan luyen.
10. Chay test unit, concurrency, restart va browser end-to-end.

Moi buoc phai giu cac test cu dang pass. Khong sua test chi de lam cho implementation sai duoc pass.


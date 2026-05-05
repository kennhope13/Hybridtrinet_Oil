@echo off
setlocal
chcp 65001 >nul

echo ======================================================
echo           OIL FORECAST - AUTOMATED HUB
echo ======================================================
echo.

:: Chuyen den thu muc chua file bat
cd /d "%~dp0"

:: Khoi tao file log neu chua co
if not exist "sim_log.txt" (
    echo [INFO] Khoi tao file nhat ky sim_log.txt...
    echo === NHAT KY MO PHONG GIA DAU === > sim_log.txt
)

:: 1. Tao moi truong ao neu chua ton tai
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Dang tao moi truong ao (venv)...
    py -3.11 -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Khong the tao venv. Vui long kiem tra Python 3.11.
        pause
        exit /b
    )
)

:: 2. Cap nhat pip va cai dat thu vien
echo [INFO] Dang kiem tra va cap nhat thu vien...
"venv\Scripts\python.exe" -m pip install --upgrade pip -q
"venv\Scripts\python.exe" -m pip install -r requirements.txt -q

:: Go bo PyArrow neu co de tranh loi DLL
"venv\Scripts\python.exe" -m pip uninstall -y pyarrow >nul 2>&1

:: 3. Kiem tra xem da co checkpoint chua
::    Chi train lan dau neu CHUA co bat ky checkpoint nao
echo.
echo [INFO] Kiem tra checkpoint mo hinh...

set NEED_TRAIN=0
if not exist "checkpoints_multi\gumnet_h1.pt"  set NEED_TRAIN=1
if not exist "checkpoints_multi\gumnet_h5.pt"  set NEED_TRAIN=1
if not exist "checkpoints_multi\gumnet_h10.pt" set NEED_TRAIN=1
if not exist "checkpoints_multi\gumnet_h30.pt" set NEED_TRAIN=1
if not exist "checkpoints_multi\gumnet_h60.pt" set NEED_TRAIN=1

if %NEED_TRAIN%==1 (
    echo.
    echo ======================================================
    echo  BUOC KHOI TAO: HUAN LUYEN MO HINH TU DU LIEU GOC
    echo  GUMNet + HybridTriNet -- Ca 5 moc: 1, 5, 10, 30, 60 ngay
    echo  Qua trinh nay chi chay MOT LAN DUY NHAT.
    echo  Vui long cho... co the mat 10-30 phut.
    echo ======================================================
    echo.
    "venv\Scripts\python.exe" train_all_horizons.py --horizons 1 5 10 30 60 --models GUMNet HybridTriNet
    if %errorlevel% neq 0 (
        echo [WARNING] Co loi khi huan luyen. Ung dung van se chay, ket qua co the han che.
    ) else (
        echo.
        echo [OK] Huan luyen hoan tat! Tat ca checkpoint da san sang.
    )
) else (
    echo [OK] Da co du checkpoint. Bo qua buoc huan luyen.
)

:: 4. Chay ung dung Streamlit
echo.
echo [INFO] Dang khoi dong giao dien...
echo [INFO] Mo trinh duyet tai: http://localhost:8502
echo.
"venv\Scripts\python.exe" -m streamlit run app_main.py --server.port 8502

pause

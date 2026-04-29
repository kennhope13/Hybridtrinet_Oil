@echo off
setlocal

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

:: 1. Tao moi truong ao neu chua ton tai (Su dung Python 3.11)
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Dang tao moi truong ao (venv) voi Python 3.11...
    py -3.11 -m venv venv
    if %errorlevel% neq 0 (
        echo [ERROR] Khong the tao venv. Vui long kiem tra xem may da cai Python 3.11 chua.
        pause
        exit /b
    )
)

:: 2. Cap nhat pip va cai dat thu vien
echo [INFO] Dang kiem tra va cai dat thu vien...
"venv\Scripts\python.exe" -m pip install --upgrade pip
"venv\Scripts\python.exe" -m pip install -r requirements.txt

:: Go bo PyArrow neu co de tranh loi DLL Blocked tren mot so may
"venv\Scripts\python.exe" -m pip uninstall -y pyarrow >nul 2>&1

:: 3. Chay ung dung Streamlit
echo [INFO] Dang chay ung dung...
"venv\Scripts\python.exe" -m streamlit run app_main.py --server.port 8502

pause

@echo off
setlocal
chcp 65001 >nul

echo ======================================================
echo           OIL FORECAST - AUTOMATED HUB
echo ======================================================
echo.

cd /d "%~dp0"

if not exist "sim_log.txt" echo [INFO] Khoi tao file sim_log.txt... & echo === NHAT KY MO PHONG GIA DAU === > sim_log.txt

:: 1. Kiem tra venv
if not exist "venv\Scripts\python.exe" (
    echo [INFO] Dang tao moi truong ao (venv)...
    python -m venv venv
)

if not exist "venv\Scripts\python.exe" (
    echo [ERROR] Khong tim thay Python. Vui long cai dat Python va thu lai.
    pause
    exit /b
)

:: 2. Cai dat thu vien
echo [INFO] Dang kiem tra thu vien...
"venv\Scripts\python.exe" -m pip install --upgrade pip -q
"venv\Scripts\python.exe" -m pip install -r requirements.txt -q
"venv\Scripts\python.exe" -m pip uninstall -y pyarrow >nul 2>&1

:: 3. Chay ung dung
echo.
echo [INFO] Dang khoi dong giao dien...
echo [INFO] Mo trinh duyet tai: http://localhost:8502
echo.

"venv\Scripts\python.exe" -m streamlit run app_main.py --server.port 8502

pause

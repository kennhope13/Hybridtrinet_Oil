@echo off
title Oil Forecast Evaluation Hub
echo ======================================================
echo           OIL FORECAST - AUTOMATED HUB
echo ======================================================
echo.

:: 1. Kiem tra va tao Moi truong ao (Virtual Environment) de tranh xung dot
set VENV_PATH=%~dp0venv
if not exist "%VENV_PATH%" (
    echo [!] Dang khoi tao moi truong co lap (Virtual Environment)...
    echo     (Viec nay giup tranh xung dot voi cac phan mem khac tren may ban)
    python -m venv venv
    if %errorlevel% neq 0 (
        echo [X] LOI: Khong the khoi tao moi truong ao. Vui long kiem tra lai Python.
        pause
        exit /b
    )
    echo [OK] Da tao xong moi truong co lap.
)

:: 2. Kich hoat moi truong ao
echo [*] Dang kich hoat moi truong...
call "%VENV_PATH%\Scripts\activate"

:: 3. Tu dong cap nhat va sua loi thu vien
echo [*] Dang kiem tra va tu dong sua loi thu vien (neu co)...
python -m pip install --upgrade pip >nul
pip install -r requirements.txt --quiet
if %errorlevel% neq 0 (
    echo [X] LOI: Khong the cap nhat thu vien. Vui long kiem tra ket noi mang.
    pause
    exit /b
)

echo.
echo [OK] MOI TRUONG DA SAN SANG! 
echo     (Moi thu vien da duoc co lap, khong lo xung dot)
echo.
echo 4. Dang khoi dong ung dung...
streamlit run app_main.py --server.port 8502
pause

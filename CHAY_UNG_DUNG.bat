@echo off
setlocal
chcp 65001 >nul

echo ======================================================
echo           OIL FORECAST - AUTOMATED HUB
echo ======================================================
echo.

:: Chuyen den thu muc chua file bat
cd /d "%~dp0"

:: 1. Kiem tra venv (Dung goto de tranh loi ngoac don voi duong dan co dau cach)
if exist "venv\Scripts\python.exe" goto :VENV_EXIST

echo [INFO] Dang tao moi truong ao (venv)...
python -m venv venv
if errorlevel 1 goto :ERROR_PYTHON

:VENV_EXIST
:: 2. Cap nhat thu vien
echo [INFO] Dang kiem tra va cap nhat thu vien...
"venv\Scripts\python.exe" -m pip install --upgrade pip -q
if errorlevel 1 goto :ERROR_PIP

"venv\Scripts\python.exe" -m pip install -r requirements.txt -q
"venv\Scripts\python.exe" -m pip uninstall -y pyarrow >nul 2>&1

:: 3. Chay ung dung
echo.
echo [INFO] Dang khoi dong giao dien...
echo [INFO] Mo trinh duyet tai: http://localhost:8502
echo.

"venv\Scripts\python.exe" -m streamlit run app_main.py --server.port 8502
goto :END

:ERROR_PYTHON
echo.
echo [ERROR] Khong tim thay Python hoac khong the tao venv.
echo Vui long kiem tra xem Python da duoc cai dat va them vao PATH chua.
pause
goto :END

:ERROR_PIP
echo.
echo [ERROR] Co loi khi cai dat thu vien. 
echo Vui long kiem tra ket noi mang va thu lai.
pause
goto :END

:END
pause

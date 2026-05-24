@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul

echo ======================================================
echo           OIL FORECAST - AUTOMATED HUB
echo ======================================================
echo.

:: Chuyen den thu muc chua file bat
cd /d "%~dp0"

:: 1. Kiem tra xem Python da duoc cai dat chua
where python >nul 2>&1
if %errorlevel% neq 0 (
    echo [WARNING] Khong tim thay Python tren he thong.
    echo [INFO] Dang tu dong tai va cai dat Python 3.11.9 - phien ban Silent...
    
    rem Tai python ve thu muc Temp
    powershell -Command "[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe' -OutFile '%temp%\python_installer.exe'"
    if errorlevel 1 (
        echo [ERROR] Khong the tai xuong Python. Vui long kiem tra ket noi internet.
        pause
        exit /b 1
    )
    
    echo [INFO] Dang cai dat Python ngam...
    rem Cai dat cho rieng user hien tai khong can quyen Admin, tu dong add vao PATH
    start /wait "" "%temp%\python_installer.exe" /quiet InstallAllUsers=0 PrependPath=1 Include_test=0 Include_pip=1
    del "%temp%\python_installer.exe"
    
    rem Thiet lap PATH tam thoi cho cmd hien tai
    set "PYTHON_PATH=%LocalAppData%\Programs\Python\Python311"
    set "PATH=!PYTHON_PATH!;!PYTHON_PATH!\Scripts;!PATH!"
    
    echo [SUCCESS] Da tu dong cai dat Python thanh cong!
)

:: 2. Kiem tra venv
if exist "venv\Scripts\python.exe" goto :VENV_EXIST

echo [INFO] Dang tao moi truong ao - venv...
python -m venv venv
if errorlevel 1 (
    rem Neu python he thong bi loi, thu goi truc tiep tu LocalAppData
    "%LocalAppData%\Programs\Python\Python311\python.exe" -m venv venv
)
if errorlevel 1 goto :ERROR_PYTHON

:VENV_EXIST
:: 3. Cap nhat thu vien
echo [INFO] Dang kiem tra va cap nhat thu vien...
"venv\Scripts\python.exe" -m pip install --upgrade pip -q
if errorlevel 1 goto :ERROR_PIP

rem Kiem tra xem may co NVIDIA GPU hay khong
nvidia-smi >nul 2>&1
if %errorlevel% equ 0 (
    echo [INFO] Phat hien NVIDIA GPU. Dang cai dat PyTorch ho tro GPU CUDA...
    "venv\Scripts\python.exe" -m pip install torch --index-url https://download.pytorch.org/whl/cu121 -q
) else (
    echo [INFO] Khong tim thay NVIDIA GPU hoac Driver CUDA. Su dung PyTorch phien ban CPU...
)

"venv\Scripts\python.exe" -m pip install -r requirements.txt -q
"venv\Scripts\python.exe" -m pip install pyarrow -q

:: 4. Chay ung dung
echo.
echo [INFO] Dang khoi dong giao dien...
echo [INFO] Mo trinh duyet tai: http://localhost:8502
echo.

"venv\Scripts\python.exe" -m streamlit run app_main.py --server.port 8502
goto :END

:ERROR_PYTHON
echo.
echo [ERROR] Khong the khoi tao moi truong ao Python.
pause
goto :END

:ERROR_PIP
echo.
echo [ERROR] Co loi khi cai dat thu vien. 
pause
goto :END

:END
pause

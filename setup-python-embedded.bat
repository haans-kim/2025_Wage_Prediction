@echo off
REM =========================================================
REM Python 3.10 Embedded 다운로드 및 설정 스크립트
REM =========================================================

echo.
echo ========================================
echo Python Embedded Setup Script
echo ========================================
echo.

set PYTHON_VERSION=3.10.11
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip
set DOWNLOAD_DIR=resources\python-runtime
set ZIP_FILE=python-embedded.zip

REM 디렉토리 생성
echo [1/6] Creating directories...
if not exist "%DOWNLOAD_DIR%" mkdir "%DOWNLOAD_DIR%"
echo Created: %DOWNLOAD_DIR%
echo.

REM Python Embedded 다운로드
echo [2/6] Downloading Python %PYTHON_VERSION% Embedded...
echo URL: %PYTHON_URL%
echo.
powershell -Command "& {Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%ZIP_FILE%'}"
if errorlevel 1 (
    echo ERROR: Failed to download Python Embedded
    pause
    exit /b 1
)
echo Download complete.
echo.

REM 압축 해제
echo [3/6] Extracting Python Embedded...
powershell -Command "& {Expand-Archive -Path '%ZIP_FILE%' -DestinationPath '%DOWNLOAD_DIR%' -Force}"
if errorlevel 1 (
    echo ERROR: Failed to extract Python Embedded
    pause
    exit /b 1
)
echo Extraction complete.
echo.

REM 다운로드 파일 삭제
echo [4/6] Cleaning up...
del /Q "%ZIP_FILE%"
echo Removed temporary zip file.
echo.

REM python3xx._pth 파일 수정 (site-packages 활성화)
echo [5/6] Configuring Python paths...
set PTH_FILE=%DOWNLOAD_DIR%\python310._pth

REM 기존 파일 백업
if exist "%PTH_FILE%" (
    copy /Y "%PTH_FILE%" "%PTH_FILE%.bak" >nul
)

REM 새로운 _pth 파일 작성
(
echo python310.zip
echo .
echo Lib
echo Lib\site-packages
echo.
echo # Uncomment to run site.main automatically
echo import site
) > "%PTH_FILE%"

echo Updated: %PTH_FILE%
echo.

REM get-pip.py 다운로드 (선택사항)
echo [6/6] Downloading get-pip.py...
powershell -Command "& {Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile '%DOWNLOAD_DIR%\get-pip.py'}"
if errorlevel 1 (
    echo WARNING: Failed to download get-pip.py
    echo You can download it manually if needed.
) else (
    echo Downloaded: get-pip.py
)
echo.

echo ========================================
echo SETUP COMPLETE!
echo ========================================
echo.
echo Python Embedded installed at: %DOWNLOAD_DIR%
echo.
echo Next steps:
echo 1. Copy site-packages from venv:
echo    xcopy /E /I /Y backend\venv\Lib\site-packages %DOWNLOAD_DIR%\Lib\site-packages
echo.
echo 2. Test Python:
echo    %DOWNLOAD_DIR%\python.exe --version
echo.
echo 3. Test imports:
echo    %DOWNLOAD_DIR%\python.exe -c "import pycaret; print('PyCaret OK')"
echo.

pause

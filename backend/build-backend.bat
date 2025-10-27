@echo off
REM =========================================================
REM SambioWage Backend Build Script
REM FastAPI 서버를 PyInstaller로 단일 실행 파일로 번들링
REM =========================================================

echo.
echo ========================================
echo SambioWage Backend Build Script
echo ========================================
echo.

REM 현재 디렉토리 확인
echo [1/6] Checking current directory...
cd /d %~dp0
echo Current directory: %CD%
echo.

REM 가상환경 확인
echo [2/6] Checking virtual environment...
if not exist "venv\Scripts\python.exe" (
    echo ERROR: Virtual environment not found!
    echo Please create it first: py -3.10 -m venv venv
    pause
    exit /b 1
)
echo Virtual environment found.
echo.

REM 가상환경 활성화
echo [3/6] Activating virtual environment...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo ERROR: Failed to activate virtual environment
    pause
    exit /b 1
)
echo Activated.
echo.

REM PyInstaller 설치 확인 및 설치
echo [4/6] Checking PyInstaller...
python -m pip show pyinstaller >nul 2>&1
if errorlevel 1 (
    echo PyInstaller not found. Installing...
    python -m pip install pyinstaller
    if errorlevel 1 (
        echo ERROR: Failed to install PyInstaller
        pause
        exit /b 1
    )
) else (
    echo PyInstaller is already installed.
)
echo.

REM 이전 빌드 결과 삭제
echo [5/6] Cleaning previous build...
if exist "dist" (
    echo Removing old dist folder...
    rmdir /s /q dist
)
if exist "build" (
    echo Removing old build folder...
    rmdir /s /q build
)
echo.

REM PyInstaller로 빌드
echo [6/6] Building with PyInstaller...
echo This may take 5-10 minutes...
echo.
pyinstaller backend-server.spec

if errorlevel 1 (
    echo.
    echo ========================================
    echo BUILD FAILED!
    echo ========================================
    echo Please check the error messages above.
    pause
    exit /b 1
)

echo.
echo ========================================
echo BUILD SUCCESS!
echo ========================================
echo.
echo Build output: %CD%\dist\backend-server\
echo Executable: %CD%\dist\backend-server\backend-server.exe
echo.
echo Next steps:
echo 1. Test the executable: cd dist\backend-server ^& backend-server.exe
echo 2. Check http://localhost:8000 in your browser
echo 3. Test API endpoints at http://localhost:8000/docs
echo.

pause

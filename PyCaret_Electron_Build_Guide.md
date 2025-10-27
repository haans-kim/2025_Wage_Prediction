# PyCaret 프로젝트 스탠드얼론 데스크톱 앱 빌드 가이드

> **프로젝트**: SambioWage 2025 (React + FastAPI + PyCaret + Electron)
> **목표**: 웹 애플리케이션을 설치 없이 실행 가능한 단일 .exe 파일로 배포
> **작성일**: 2025-10-27

---

## 목차
1. [개요](#1-개요)
2. [아키텍처 결정](#2-아키텍처-결정)
3. [환경 설정](#3-환경-설정)
4. [백엔드 빌드 (Python Embedded)](#4-백엔드-빌드-python-embedded)
5. [프론트엔드 빌드](#5-프론트엔드-빌드)
6. [Electron 통합](#6-electron-통합)
7. [최종 패키징](#7-최종-패키징)
8. [배포 패키지 생성](#8-배포-패키지-생성)
9. [문제 해결](#9-문제-해결)
10. [체크리스트](#10-체크리스트)

---

## 1. 개요

### 1.1 기술 스택
- **Frontend**: React + TypeScript + Tailwind CSS
- **Backend**: FastAPI + Python 3.10
- **ML Framework**: PyCaret (scikit-learn, lightgbm, xgboost 등 포함)
- **Desktop Wrapper**: Electron 25.9.8
- **Python Distribution**: Python Embedded 3.10.11

### 1.2 왜 Python Embedded를 선택했나?
- ❌ **PyInstaller 실패**: PyCaret의 복잡한 의존성으로 인해 빌드 실패
  - scikit-learn, lightgbm, xgboost 등 C 확장 모듈 충돌
  - 동적 import로 인한 모듈 누락
- ✅ **Python Embedded 성공**:
  - 전체 Python 런타임 포함
  - 모든 의존성 보장
  - 안정적인 실행

### 1.3 최종 결과물
- **단일 실행 파일**: `SambioWage.exe` (392MB)
- **포함 사항**:
  - Electron 런타임
  - Python 3.10.11 Embedded (800MB+)
  - FastAPI 백엔드
  - React 프론트엔드
  - PyCaret + 모든 ML 라이브러리
  - 초기 데이터 및 모델

---

## 2. 아키텍처 결정

### 2.1 빌드 전략 비교

| 방법 | 장점 | 단점 | 결과 |
|------|------|------|------|
| PyInstaller | 작은 파일 크기 | PyCaret 의존성 문제 | ❌ 실패 |
| Python Embedded | 안정성, 모든 의존성 보장 | 큰 파일 크기 (800MB+) | ✅ 성공 |
| Docker | 격리된 환경 | Windows에서 설치 필요 | ❌ 요구사항 불충족 |

### 2.2 선택한 아키텍처

```
SambioWage.exe
├── Electron Main Process
│   ├── Backend Launcher (Python Embedded + FastAPI)
│   │   └── http://localhost:8000
│   ├── Frontend Server (Express + React Build)
│   │   └── http://localhost:3000
│   └── BrowserWindow
│       └── Loads http://localhost:3000
```

### 2.3 핵심 설계 원칙
1. **ASAR 비활성화**: Python 런타임과 FastAPI가 파일 시스템 접근 필요
2. **절대 경로 사용**: Electron 환경에서 경로 문제 방지
3. **환경 변수 전달**: `ELECTRON_APP=true`, `APP_DATA_DIR` 설정
4. **프로세스 관리**: Electron 종료 시 Python 프로세스도 자동 종료

---

## 3. 환경 설정

### 3.1 필수 도구 설치

```bash
# Node.js 18+ 설치
node --version  # v18.0.0+
npm --version   # v9.0.0+

# Python 3.10 설치 (개발용)
python --version  # Python 3.10.x

# Git 설치
git --version
```

### 3.2 프로젝트 구조

```
project/
├── backend/                 # FastAPI 백엔드
│   ├── app/
│   │   ├── api/
│   │   ├── services/
│   │   └── core/
│   │       └── config.py    # ⚠️ 경로 설정 중요!
│   ├── data/                # 초기 데이터
│   ├── models/              # 학습된 모델
│   ├── requirements.txt
│   └── run.py
├── frontend/                # React 프론트엔드
│   ├── src/
│   ├── public/
│   └── package.json
├── electron/                # Electron 메인 프로세스
│   ├── main.ts              # ⚠️ 핵심 파일!
│   ├── preload.ts
│   └── tsconfig.json
├── resources/               # 빌드 리소스
│   └── python-runtime/      # Python Embedded (빌드 시)
├── package.json             # ⚠️ electron-builder 설정
└── .gitignore
```

---

## 4. 백엔드 빌드 (Python Embedded)

### 4.1 Python Embedded 다운로드

**자동화 스크립트**: `setup-python-embedded.bat`

```batch
@echo off
echo ========================================
echo Python Embedded Setup for Electron
echo ========================================

set PYTHON_VERSION=3.10.11
set PYTHON_URL=https://www.python.org/ftp/python/%PYTHON_VERSION%/python-%PYTHON_VERSION%-embed-amd64.zip
set DOWNLOAD_PATH=python-embedded.zip
set EXTRACT_PATH=resources\python-runtime

echo Downloading Python Embedded %PYTHON_VERSION%...
powershell -Command "Invoke-WebRequest -Uri '%PYTHON_URL%' -OutFile '%DOWNLOAD_PATH%'"

echo Extracting to %EXTRACT_PATH%...
if not exist %EXTRACT_PATH% mkdir %EXTRACT_PATH%
powershell -Command "Expand-Archive -Path '%DOWNLOAD_PATH%' -DestinationPath '%EXTRACT_PATH%' -Force"

echo Installing pip...
cd %EXTRACT_PATH%
echo import sys; sys.path.insert(0, '') > python310._pth
powershell -Command "Invoke-WebRequest -Uri 'https://bootstrap.pypa.io/get-pip.py' -OutFile 'get-pip.py'"
python.exe get-pip.py

echo Installing dependencies...
python.exe -m pip install --upgrade pip
python.exe -m pip install -r ..\..\backend\requirements.txt

echo Cleanup...
del get-pip.py
cd ..\..
del %DOWNLOAD_PATH%

echo ========================================
echo Python Embedded Setup Complete!
echo ========================================
pause
```

### 4.2 실행 방법

```bash
# 프로젝트 루트에서 실행
cmd /c setup-python-embedded.bat
```

### 4.3 결과 확인

```bash
ls resources/python-runtime/
# python.exe, python310.dll, Lib/, site-packages/ 등 확인
# 크기: 약 800MB+

# 테스트
resources/python-runtime/python.exe -c "import pycaret; print('OK')"
```

### 4.4 핵심 설정: `backend/app/core/config.py`

⚠️ **가장 중요한 파일!** 경로 설정 문제의 90%가 여기서 발생합니다.

```python
from pydantic_settings import BaseSettings
from typing import List
import os
from pathlib import Path

def get_app_data_dir() -> Path:
    """
    애플리케이션 데이터 디렉토리를 반환합니다.
    Electron 환경에서는 실행 파일 위치를 기준으로 합니다.
    """
    if os.getenv('ELECTRON_APP') == 'true':
        # ✅ Electron 환경: APP_DATA_DIR 환경 변수 사용
        app_dir = Path(os.getenv('APP_DATA_DIR', '.'))
    else:
        # 개발 환경: 프로젝트 루트
        app_dir = Path(__file__).parent.parent.parent

    return app_dir.absolute()

# ✅ 절대 경로 사용!
BASE_DIR = get_app_data_dir()
UPLOAD_DIR = BASE_DIR / 'uploads'
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# 디렉토리 자동 생성
for dir_path in [UPLOAD_DIR, DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SambioWage"
    VERSION: str = "1.0.0"

    # ✅ 절대 경로를 문자열로 변환
    UPLOAD_DIR: str = str(UPLOAD_DIR)
    DATA_DIR: str = str(DATA_DIR)
    MODEL_DIR: str = str(MODELS_DIR)

    # CORS 설정
    ALLOWED_HOSTS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "*"
    ]

settings = Settings()
```

### 4.5 서비스에서 경로 사용: `backend/app/services/data_service.py`

⚠️ **흔한 실수**: `Path("data")` 같은 상대 경로 사용 금지!

```python
import pandas as pd
from pathlib import Path
from app.core.config import settings, UPLOAD_DIR, DATA_DIR  # ✅ import!

class DataService:
    def __init__(self):
        # ✅ config.py에서 가져온 절대 경로 사용
        self.upload_dir = Path(UPLOAD_DIR)
        self.data_dir = Path(DATA_DIR)

        # ❌ 이렇게 하면 안 됨!
        # self.data_dir = Path("data")  # 상대 경로 - 오류 원인!

        self.pickle_file = self.data_dir / "master_data.pkl"
        self.working_pickle_file = self.data_dir / "working_data.pkl"
```

---

## 5. 프론트엔드 빌드

### 5.1 환경 변수 설정

⚠️ **중요**: Electron용 빌드는 localhost API URL 사용!

**`.env.production` (Railway 배포용)**
```env
REACT_APP_API_URL=https://sambiowage-backend.up.railway.app
```

**Electron용 빌드 명령**
```bash
cd frontend

# ✅ 환경 변수 오버라이드
export REACT_APP_API_URL=http://localhost:8000
npm run build

# Windows에서는
set REACT_APP_API_URL=http://localhost:8000
npm run build
```

### 5.2 API 클라이언트 설정: `frontend/src/lib/api.ts`

```typescript
const API_BASE_URL = process.env.REACT_APP_API_URL || 'http://localhost:8000';

class ApiClient {
  private baseUrl: string;

  constructor() {
    this.baseUrl = API_BASE_URL;
    console.log('API Base URL:', this.baseUrl);  // 디버깅용
  }

  async uploadData(file: File): Promise<DataUploadResponse> {
    const formData = new FormData();
    formData.append('file', file);

    const response = await fetch(`${this.baseUrl}/api/data/upload`, {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }

    return response.json();
  }
}

export const apiClient = new ApiClient();
```

### 5.3 빌드 확인

```bash
ls frontend/build/
# index.html, static/, manifest.json 등 확인
```

---

## 6. Electron 통합

### 6.1 TypeScript 설정: `electron/tsconfig.json`

```json
{
  "compilerOptions": {
    "target": "ES2020",
    "module": "commonjs",
    "lib": ["ES2020"],
    "outDir": "../dist-electron",
    "rootDir": ".",
    "strict": true,
    "esModuleInterop": true,
    "skipLibCheck": true,
    "forceConsistentCasingInFileNames": true,
    "moduleResolution": "node",
    "resolveJsonModule": true,
    "types": ["node", "electron"]
  },
  "include": ["**/*.ts"],
  "exclude": ["node_modules"]
}
```

### 6.2 메인 프로세스: `electron/main.ts`

⚠️ **핵심 파일 - 자세한 주석 포함**

```typescript
import { app, BrowserWindow, dialog } from 'electron';
import * as path from 'path';
import { spawn } from 'child_process';
import * as fs from 'fs';
import express from 'express';

let mainWindow: BrowserWindow | null = null;
let backendProcess: any = null;
let expressApp: any = null;
let expressServer: any = null;

// ========================================
// 1. 경로 설정
// ========================================

function getAppDataDir(): string {
  if (app.isPackaged) {
    // ✅ 패키징된 경우: exe 파일이 있는 디렉토리
    // 예: C:\Users\...\release\win-unpacked\
    return path.dirname(process.execPath);
  } else {
    // 개발 모드: 프로젝트 루트
    return process.cwd();
  }
}

const appDataDir = getAppDataDir();
const logFile = path.join(appDataDir, 'electron-main.log');

// 로깅 함수
function log(message: string) {
  const timestamp = new Date().toISOString();
  const logMessage = `[${timestamp}] ${message}\n`;
  console.log(message);
  fs.appendFileSync(logFile, logMessage);
}

log('=== Electron Main Process Started ===');
log(`isDev: ${!app.isPackaged}`);
log(`process.execPath: "${process.execPath}"`);
log(`App Data Dir: ${appDataDir}`);

// ========================================
// 2. Window 생성
// ========================================

function createWindow() {
  log('Creating window...');

  mainWindow = new BrowserWindow({
    width: 1400,
    height: 900,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      preload: path.join(__dirname, 'preload.js'),
    },
    show: false,  // 서버 준비될 때까지 숨김
  });

  mainWindow.once('ready-to-show', () => {
    log('Window ready to show');
    mainWindow?.show();
  });

  mainWindow.on('closed', () => {
    mainWindow = null;
  });
}

// ========================================
// 3. Backend (Python) 시작
// ========================================

async function startBackend() {
  log('=== Starting Backend (Python FastAPI) ===');

  // 3.1 Python 경로
  const pythonExe = app.isPackaged
    ? path.join(process.resourcesPath, 'python-runtime', 'python.exe')
    : path.join(appDataDir, 'resources', 'python-runtime', 'python.exe');

  // 3.2 Backend 스크립트 경로
  const backendScript = app.isPackaged
    ? path.join(process.resourcesPath, 'app', 'backend', 'run.py')
    : path.join(appDataDir, 'backend', 'run.py');

  const backendDir = path.dirname(backendScript);

  log(`Python exe: "${pythonExe}"`);
  log(`Backend script: "${backendScript}"`);
  log(`Backend dir: "${backendDir}"`);

  // 3.3 파일 존재 확인
  if (!fs.existsSync(pythonExe)) {
    const msg = `Python executable not found at: ${pythonExe}`;
    log(`ERROR: ${msg}`);
    dialog.showErrorBox('Python Error', msg);
    app.quit();
    return;
  }

  if (!fs.existsSync(backendScript)) {
    const msg = `Backend script not found at: ${backendScript}`;
    log(`ERROR: ${msg}`);
    dialog.showErrorBox('Backend Error', msg);
    app.quit();
    return;
  }

  log('Python exists: true');
  log('Backend script exists: true');

  // 3.4 환경 변수 설정 ⚠️ 중요!
  const env = {
    ...process.env,
    ELECTRON_APP: 'true',              // ✅ Electron 환경 표시
    APP_DATA_DIR: appDataDir,          // ✅ 데이터 디렉토리
    PORT: '8000',
    PYTHONPATH: backendDir,
  };

  log('Starting backend process...');
  log('Environment:', JSON.stringify(env, null, 2));

  const backendLogFile = path.join(appDataDir, 'backend.log');

  // 3.5 Python 프로세스 시작
  backendProcess = spawn(pythonExe, [backendScript], {
    cwd: backendDir,
    env: env,
    shell: false,
    windowsHide: false,  // 디버깅용
  });

  // 3.6 로그 처리
  backendProcess.stdout.on('data', (data: any) => {
    const message = data.toString();
    log(`[Backend STDOUT] ${message}`);
    fs.appendFileSync(backendLogFile, message);
  });

  backendProcess.stderr.on('data', (data: any) => {
    const message = data.toString();
    log(`[Backend STDERR] ${message}`);
    fs.appendFileSync(backendLogFile, message);
  });

  // 3.7 프로세스 에러 처리
  backendProcess.on('error', (error: any) => {
    log(`Backend process error: ${error.message}`);
    dialog.showErrorBox('Backend Error',
      `Failed to start backend.\n\nError: ${error.message}`);
  });

  // 3.8 프로세스 종료 처리
  backendProcess.on('exit', (code: any, signal: any) => {
    log(`Backend exited with code ${code}, signal ${signal}`);
    if (code !== 0 && code !== null) {
      dialog.showErrorBox('Backend Crashed',
        `Backend stopped unexpectedly.\n\nExit code: ${code}\nSignal: ${signal}`);
    }
  });

  log('Backend process spawned successfully');
}

// ========================================
// 4. Frontend (Express) 시작
// ========================================

function startFrontend() {
  log('=== Starting Frontend (Express) ===');

  // 4.1 React 빌드 경로
  const buildPath = app.isPackaged
    ? path.join(process.resourcesPath, 'app', 'frontend', 'build')
    : path.join(appDataDir, 'frontend', 'build');

  log(`Frontend build path: "${buildPath}"`);

  if (!fs.existsSync(buildPath)) {
    const msg = `Frontend build not found at: ${buildPath}`;
    log(`ERROR: ${msg}`);
    dialog.showErrorBox('Frontend Error', msg);
    app.quit();
    return;
  }

  log('Build path exists: true');

  // 4.2 Express 서버 설정
  expressApp = express();
  expressApp.use(express.static(buildPath));

  expressApp.get('*', (req: any, res: any) => {
    res.sendFile(path.join(buildPath, 'index.html'));
  });

  // 4.3 서버 시작
  const PORT = 3000;
  expressServer = expressApp.listen(PORT, 'localhost', () => {
    log(`Frontend server started on http://localhost:${PORT}`);
  });
}

// ========================================
// 5. 서버 대기 및 로딩
// ========================================

async function waitForServers() {
  log('Loading application...');

  // 서버 시작까지 대기 (3초면 충분)
  const waitTime = 3000;
  log(`Waiting ${waitTime}ms for servers to initialize...`);
  await new Promise(resolve => setTimeout(resolve, waitTime));

  // BrowserWindow에 로드
  if (mainWindow && !mainWindow.isDestroyed()) {
    log('Loading URL: http://localhost:3000');
    try {
      await mainWindow.loadURL('http://localhost:3000');
      log('✓ URL loaded successfully');
      return;
    } catch (error: any) {
      log(`Error loading URL: ${error.message}`);
      dialog.showErrorBox('Loading Error',
        `Failed to load application.\n\nError: ${error.message}`);
      app.quit();
    }
  }
}

// ========================================
// 6. 앱 라이프사이클
// ========================================

app.on('ready', async () => {
  log('App ready, creating window...');
  createWindow();
  startBackend();
  startFrontend();
  await waitForServers();
});

app.on('window-all-closed', () => {
  log('All windows closed');
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

app.on('before-quit', () => {
  log('Before quit');
  log('Cleaning up processes...');

  // Backend 종료
  if (backendProcess && !backendProcess.killed) {
    log('Killing backend process...');
    backendProcess.kill();
  }

  // Frontend 서버 종료
  if (expressServer) {
    log('Closing frontend server...');
    expressServer.close();
  }

  log('Cleanup complete');
});

app.on('will-quit', () => {
  log('Will quit');
});

app.on('quit', () => {
  log('App quit');
});
```

### 6.3 Preload 스크립트: `electron/preload.ts`

```typescript
// 현재는 사용하지 않지만, 나중에 IPC 통신 시 필요
import { contextBridge } from 'electron';

contextBridge.exposeInMainWorld('electron', {
  // 필요한 API 노출
});
```

---

## 7. 최종 패키징

### 7.1 루트 `package.json` 설정

⚠️ **핵심 설정 파일**

```json
{
  "name": "sambiowage",
  "version": "1.0.0",
  "description": "SambioWage - ML-based Wage Prediction Desktop Application",
  "main": "dist-electron/main.js",
  "author": "Sambio",
  "license": "MIT",
  "private": true,
  "scripts": {
    "electron:compile": "tsc -p electron/tsconfig.json",
    "electron:dev": "npm run electron:compile && cross-env NODE_ENV=development electron .",
    "electron:build:win": "npm run electron:compile && electron-builder --win --x64",
    "electron:build:mac": "npm run electron:compile && electron-builder --mac",
    "electron:build:linux": "npm run electron:compile && electron-builder --linux"
  },
  "dependencies": {
    "express": "^4.18.2"
  },
  "devDependencies": {
    "@types/electron": "^1.4.38",
    "@types/express": "^4.17.21",
    "@types/node": "^20.11.0",
    "cross-env": "^7.0.3",
    "electron": "^25.9.8",
    "electron-builder": "^24.9.1",
    "typescript": "^5.3.3"
  },
  "build": {
    "appId": "com.sambio.wage",
    "productName": "SambioWage",
    "asar": false,  // ⚠️ 중요: Python 런타임 때문에 false!
    "directories": {
      "output": "release"
    },
    "files": [
      "dist-electron/**/*",
      "frontend/build/**/*",
      "backend/app/**/*",
      "backend/data/**/*",  // ✅ 초기 데이터 포함
      "backend/run.py",
      "backend/requirements.txt",
      "resources/python-runtime/**/*",
      "node_modules/express/**/*",
      "!backend/venv/**/*",
      "!backend/dist/**/*",
      "!backend/build/**/*",
      "!**/*.pyc",
      "!**/__pycache__",
      "!**/.git"
    ],
    "extraResources": [
      {
        "from": "resources/python-runtime",
        "to": "python-runtime"
      },
      {
        "from": "backend/data",
        "to": "data"  // ✅ exe 레벨에 데이터 복사
      },
      {
        "from": "backend/models",
        "to": "models"  // ✅ exe 레벨에 모델 복사
      }
    ],
    "win": {
      "target": [
        {
          "target": "portable",  // 단일 .exe 파일
          "arch": ["x64"]
        }
      ],
      "signAndEditExecutable": false  // ⚠️ 서명 비활성화
    },
    "mac": {
      "target": ["dmg"],
      "category": "public.app-category.business"
    },
    "linux": {
      "target": ["AppImage"],
      "category": "Office"
    }
  }
}
```

### 7.2 `.gitignore` 설정

```gitignore
# Electron
dist-electron/
release/
*.zip

# Python Embedded
resources/python-runtime/
python-embedded.zip

# Build artifacts
backend/venv/
backend/dist/
backend/build/
backend/__pycache__/
backend/*.spec

# Frontend
frontend/node_modules/
frontend/build/

# Node
node_modules/
package-lock.json

# Logs
*.log
```

### 7.3 빌드 순서

```bash
# 1. Python Embedded 설정 (최초 1회)
cmd /c setup-python-embedded.bat

# 2. Frontend 빌드
cd frontend
set REACT_APP_API_URL=http://localhost:8000
npm run build
cd ..

# 3. Electron 빌드
npm run electron:build:win
```

### 7.4 빌드 결과

```
release/
├── win-unpacked/              # 압축 해제된 버전 (테스트용)
│   ├── SambioWage.exe        # 163MB
│   ├── resources/
│   │   ├── python-runtime/   # 800MB+
│   │   └── app/
│   │       ├── backend/
│   │       └── frontend/
│   ├── data/                  # 초기 데이터
│   └── models/                # 학습된 모델
└── SambioWage 1.0.0.exe      # 392MB (portable)
```

---

## 8. 배포 패키지 생성

### 8.1 배포 파일 구조

```
SambioWage2025/
├── SambioWage.exe          # 392MB (portable exe 복사 후 이름 변경)
├── data/
│   ├── master_data.pkl     # 초기 데이터
│   └── working_data.pkl
├── models/
│   └── latest.pkl          # 학습된 모델
├── README.md               # 영문 설명서
└── 사용설명서.txt          # 한글 설명서
```

### 8.2 자동 배포 스크립트

```bash
# deploy.sh (Git Bash / WSL)
#!/bin/bash

DEPLOY_DIR="/d/SambioWage2025"
SOURCE_DIR="/c/Project/2025_Wage_Prediction"

echo "Creating deployment package..."

# 1. 폴더 생성
mkdir -p "$DEPLOY_DIR"/{data,models}

# 2. 실행 파일 복사
cp "$SOURCE_DIR/release/SambioWage 1.0.0.exe" "$DEPLOY_DIR/SambioWage.exe"

# 3. 데이터 복사
cp "$SOURCE_DIR/backend/data/"*.pkl "$DEPLOY_DIR/data/"

# 4. 모델 복사
cp "$SOURCE_DIR/backend/models/"*.pkl "$DEPLOY_DIR/models/"

# 5. 문서 복사
cp "$SOURCE_DIR/README.md" "$DEPLOY_DIR/"

echo "Deployment package created at: $DEPLOY_DIR"
ls -lh "$DEPLOY_DIR"
```

### 8.3 사용 설명서 작성

**`사용설명서.txt`** (한글)
```
===================================================================
    SambioWage 2025 - 임금인상률 예측 시스템
===================================================================

■ 실행 방법
  1. SambioWage.exe를 더블클릭
  2. 3-5초 대기 (백엔드 서버 시작)
  3. 자동으로 애플리케이션 창이 열림

■ 시스템 요구사항
  - Windows 10/11 (64-bit)
  - 최소 4GB RAM
  - 약 1GB 여유 디스크 공간

■ 주의사항
  - 포트 3000, 8000이 사용 중이면 실행 실패
  - 방화벽 경고 시 "액세스 허용" 클릭

■ 문제 해결
  [문제] "Frontend Error: address already in use"
  [해결] 다른 프로그램이 포트 사용 중 - 종료 후 재실행

  [문제] "Backend Crashed"
  [해결] backend.log 파일 확인

■ 기술 지원
  - 이메일: support@sambio.com
```

---

## 9. 문제 해결

### 9.1 데이터 파일 로딩 실패

**증상**: "저장된 데이터가 없습니다"

**원인**: 경로 문제
```python
# ❌ 잘못된 방법
self.data_dir = Path("data")  # 상대 경로

# ✅ 올바른 방법
from app.core.config import DATA_DIR
self.data_dir = Path(DATA_DIR)  # 절대 경로
```

**해결**:
1. `config.py`에서 `get_app_data_dir()` 확인
2. 환경 변수 `ELECTRON_APP=true`, `APP_DATA_DIR` 설정 확인
3. `extraResources`로 data 폴더 복사 확인

### 9.2 Windows cp949 인코딩 에러

**증상**: `UnicodeEncodeError: 'cp949' codec can't encode character`

**원인**: print()에 이모지 사용

```python
# ❌ 에러 발생
print(f"🔍 [DEBUG] Loading data...")

# ✅ 해결
print(f"[DEBUG] Loading data...")
```

### 9.3 Frontend가 Railway API 호출

**증상**: API 요청이 localhost가 아닌 Railway로 전송

**원인**: `.env.production` 사용

**해결**:
```bash
# Electron 빌드 전에 환경 변수 오버라이드
export REACT_APP_API_URL=http://localhost:8000
npm run build
```

### 9.4 Backend 프로세스가 종료되지 않음

**증상**: Electron 종료 후에도 python.exe 프로세스 남아있음

**해결**: `before-quit` 이벤트에서 명시적 종료
```typescript
app.on('before-quit', () => {
  if (backendProcess && !backendProcess.killed) {
    backendProcess.kill();
  }
});
```

### 9.5 Git Line Ending 경고

**증상**: `warning: in the working copy of 'file.json', CRLF will be replaced by LF`

**해결**:
```bash
# 1. .gitattributes 생성
*.json text eol=lf
*.ts text eol=lf
*.tsx text eol=lf
*.js text eol=lf
*.py text eol=lf

# 2. Git 설정
git config --local core.safecrlf false
```

### 9.6 electron-builder Icon 에러

**증상**: `image must be at least 256x256`

**해결**: package.json에서 icon 설정 제거 또는 올바른 크기 아이콘 사용

```json
"win": {
  // "icon": "frontend/public/logo192.png"  // 제거
}
```

---

## 10. 체크리스트

### 10.1 빌드 전 체크리스트

#### Backend
- [ ] `backend/requirements.txt` 최신 버전 확인
- [ ] `backend/app/core/config.py`에서 절대 경로 사용 확인
- [ ] 모든 서비스에서 `config.py`의 경로 import 확인
- [ ] 이모지 사용 안 함 (print, log 등)
- [ ] `backend/data/`에 초기 데이터 파일 존재 확인
- [ ] `backend/models/`에 학습된 모델 존재 확인

#### Frontend
- [ ] `npm run build` 성공 확인
- [ ] `REACT_APP_API_URL=http://localhost:8000` 설정 확인
- [ ] `frontend/build/` 폴더 생성 확인
- [ ] API 클라이언트에서 환경 변수 사용 확인

#### Electron
- [ ] `electron/main.ts` 컴파일 성공
- [ ] `package.json`의 `main` 경로 확인: `"dist-electron/main.js"`
- [ ] `asar: false` 설정 확인
- [ ] `files` 배열에 모든 필요 파일 포함 확인
- [ ] `extraResources`에 data, models 포함 확인

#### Python Embedded
- [ ] `setup-python-embedded.bat` 실행 완료
- [ ] `resources/python-runtime/python.exe` 존재 확인
- [ ] `resources/python-runtime/Lib/site-packages/pycaret` 존재 확인
- [ ] 크기 약 800MB+ 확인

### 10.2 빌드 후 체크리스트

- [ ] `release/win-unpacked/SambioWage.exe` 생성 확인
- [ ] `release/SambioWage 1.0.0.exe` 생성 확인 (portable)
- [ ] win-unpacked에서 실행 테스트
  - [ ] 창이 정상적으로 열림
  - [ ] Frontend 로딩 성공
  - [ ] API 호출 성공 (localhost:8000)
  - [ ] 데이터 로딩 성공
  - [ ] 모델 로딩 성공
- [ ] 로그 파일 확인
  - [ ] `electron-main.log` 에러 없음
  - [ ] `backend.log` 에러 없음
- [ ] 종료 테스트
  - [ ] 창 닫을 때 정상 종료
  - [ ] Python 프로세스 자동 종료 확인

### 10.3 배포 전 체크리스트

- [ ] 배포 폴더 구조 확인
- [ ] `SambioWage.exe` 복사 및 이름 변경
- [ ] `data/` 폴더 복사
- [ ] `models/` 폴더 복사
- [ ] `README.md` 작성
- [ ] `사용설명서.txt` 작성 (한글)
- [ ] 다른 PC에서 테스트
  - [ ] 실행 파일 단독 실행 확인
  - [ ] 초기 데이터 로딩 확인
  - [ ] 모든 기능 동작 확인

---

## 부록 A: 주요 명령어 모음

### 개발 환경
```bash
# Backend 실행
cd backend
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python run.py

# Frontend 실행
cd frontend
npm install
npm start

# Electron 개발 모드 (작동 안 할 수 있음 - 프로덕션 빌드 사용)
npm run electron:dev
```

### 빌드
```bash
# Python Embedded 설정
cmd /c setup-python-embedded.bat

# Frontend 빌드 (Electron용)
cd frontend
set REACT_APP_API_URL=http://localhost:8000
npm run build
cd ..

# Electron 빌드
npm run electron:build:win
```

### 테스트
```bash
# win-unpacked에서 테스트
cd release/win-unpacked
SambioWage.exe

# 로그 확인
tail -f release/win-unpacked/electron-main.log
tail -f release/win-unpacked/backend.log
```

### 배포
```bash
# 배포 폴더 생성
mkdir /d/SambioWage2025
cp release/"SambioWage 1.0.0.exe" /d/SambioWage2025/SambioWage.exe
cp -r backend/data /d/SambioWage2025/
cp -r backend/models /d/SambioWage2025/
```

---

## 부록 B: 디렉토리 구조 상세

### 개발 환경
```
project/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py
│   │   ├── api/
│   │   │   └── routes/
│   │   │       ├── data.py
│   │   │       ├── modeling.py
│   │   │       ├── analysis.py
│   │   │       └── dashboard.py
│   │   ├── services/
│   │   │   ├── data_service.py          # ⚠️ 경로 설정 주의
│   │   │   ├── modeling_service.py
│   │   │   └── analysis_service.py
│   │   └── core/
│   │       └── config.py                # ⚠️ 핵심 파일
│   ├── data/
│   │   ├── master_data.pkl
│   │   └── working_data.pkl
│   ├── models/
│   │   └── latest.pkl
│   ├── uploads/                         # 자동 생성
│   ├── requirements.txt
│   ├── run.py
│   └── venv/                            # 개발용
├── frontend/
│   ├── public/
│   ├── src/
│   │   ├── lib/
│   │   │   └── api.ts                   # ⚠️ API URL 설정
│   │   ├── pages/
│   │   ├── components/
│   │   └── App.tsx
│   ├── build/                           # npm run build 결과
│   ├── package.json
│   └── .env.production
├── electron/
│   ├── main.ts                          # ⚠️ 핵심 파일
│   ├── preload.ts
│   └── tsconfig.json
├── resources/
│   └── python-runtime/                  # Python Embedded
│       ├── python.exe
│       ├── python310.dll
│       ├── Lib/
│       └── site-packages/
│           └── pycaret/
├── dist-electron/                       # TypeScript 컴파일 결과
│   ├── main.js
│   └── preload.js
├── release/                             # 빌드 결과
│   ├── win-unpacked/
│   └── SambioWage 1.0.0.exe
├── package.json                         # ⚠️ electron-builder 설정
├── .gitignore
└── .gitattributes
```

### 패키징된 앱 (win-unpacked)
```
win-unpacked/
├── SambioWage.exe                       # Electron 실행 파일
├── resources/
│   ├── python-runtime/                  # extraResources
│   │   ├── python.exe
│   │   └── Lib/site-packages/
│   └── app/                             # files
│       ├── backend/
│       │   ├── app/
│       │   └── run.py
│       └── frontend/
│           └── build/
├── data/                                # extraResources
│   ├── master_data.pkl
│   └── working_data.pkl
├── models/                              # extraResources
│   └── latest.pkl
├── uploads/                             # 런타임에 생성
├── chrome_100_percent.pak
├── ffmpeg.dll
├── icudtl.dat
├── libEGL.dll
├── libGLESv2.dll
└── locales/
```

---

## 부록 C: 환경 변수 전달 흐름

```
Electron main.ts
    ↓ (환경 변수 설정)
    ├── ELECTRON_APP=true
    ├── APP_DATA_DIR=C:\...\win-unpacked
    └── PORT=8000
    ↓ (spawn Python process)
Python run.py
    ↓ (환경 변수 읽기)
FastAPI app (main.py)
    ↓ (import)
backend/app/core/config.py
    ↓ (get_app_data_dir() 호출)
    ├── os.getenv('ELECTRON_APP') == 'true' ✓
    ├── os.getenv('APP_DATA_DIR') → 'C:\...\win-unpacked'
    ↓
    ├── DATA_DIR = 'C:\...\win-unpacked\data'
    ├── UPLOAD_DIR = 'C:\...\win-unpacked\uploads'
    └── MODELS_DIR = 'C:\...\win-unpacked\models'
    ↓ (import)
backend/app/services/data_service.py
    ├── from app.core.config import DATA_DIR
    └── self.data_dir = Path(DATA_DIR)  # 절대 경로!
```

---

## 부록 D: 참고 링크

- **Electron 공식 문서**: https://www.electronjs.org/docs/latest/
- **electron-builder**: https://www.electron.build/
- **Python Embedded**: https://www.python.org/downloads/windows/
- **PyCaret 문서**: https://pycaret.org/
- **FastAPI 문서**: https://fastapi.tiangolo.com/

---

## 맺음말

이 가이드는 SambioWage 프로젝트를 통해 얻은 실전 경험을 바탕으로 작성되었습니다.

**핵심 포인트**:
1. **절대 경로 사용**: 모든 파일 경로는 `config.py`에서 관리
2. **환경 변수 전달**: Electron → Python으로 올바르게 전달
3. **ASAR 비활성화**: Python 런타임 때문에 필수
4. **API URL 설정**: Electron용 빌드는 localhost 사용
5. **프로세스 관리**: 종료 시 모든 프로세스 정리

다른 PyCaret 프로젝트에서도 이 가이드를 따라 스탠드얼론 앱을 만들 수 있습니다.

---

**작성**: Claude Code with Human Collaboration
**날짜**: 2025-10-27
**버전**: 1.0

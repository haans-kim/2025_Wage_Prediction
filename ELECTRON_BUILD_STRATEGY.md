# SambioWage Electron 빌드 전략 (최종)

## 전략 선택: Python Embedded + 사전 설치 패키지

PyInstaller의 복잡한 의존성 문제와 보안 환경 요구사항을 고려하여 **Python Embedded 방식**을 채택합니다.

---

## 왜 이 전략인가?

### PyInstaller 문제점
- ❌ PyCaret의 복잡한 의존성 트리로 빌드 실패
- ❌ 바이트코드 분석 에러 (IndexError: tuple index out of range)
- ❌ 빌드 시간 과다 (10-20분)
- ❌ 디버깅 어려움

### Python Embedded 장점
- ✅ **보안 환경 완벽 대응** - 인터넷 연결 불필요
- ✅ **pip install 불필요** - 모든 패키지 사전 포함
- ✅ **빌드 안정성** - 복잡한 빌드 과정 없음
- ✅ **개발 환경과 동일** - 디버깅 용이
- ✅ **유지보수 편리** - 코드 수정 시 파일만 교체

### 고객사 요구사항 충족
- ✅ 오프라인 설치 가능
- ✅ 외부 네트워크 접근 불필요
- ✅ 완전히 독립 실행 가능
- ✅ 보안 정책 준수

---

## 최종 아키텍처

```
SambioWage.exe (Electron App)
│
├── [Main Process] Electron
│   ├── 창 관리
│   ├── Backend 프로세스 시작 (python.exe)
│   └── Frontend 서빙 (Express)
│
├── [Backend Process] Python FastAPI
│   └── resources/python-runtime/python.exe run.py
│
└── [Frontend] React Static Files
    └── resources/frontend/build/
```

---

## 디렉토리 구조

```
release/win-unpacked/
├── SambioWage.exe                      # Electron 메인 실행 파일
│
├── resources/
│   ├── app/
│   │   ├── dist-electron/              # Electron 메인 프로세스
│   │   │   ├── main.js
│   │   │   └── preload.js
│   │   │
│   │   ├── frontend/build/             # React 프로덕션 빌드
│   │   │   ├── index.html
│   │   │   ├── static/
│   │   │   └── ...
│   │   │
│   │   ├── backend/                    # FastAPI 소스 코드
│   │   │   ├── app/
│   │   │   │   ├── main.py
│   │   │   │   ├── api/routes/
│   │   │   │   ├── services/
│   │   │   │   └── core/config.py
│   │   │   ├── run.py
│   │   │   └── requirements.txt
│   │   │
│   │   └── python-runtime/             # Python 임베디드 + 패키지
│   │       ├── python.exe              # Python 3.10 실행 파일
│   │       ├── python310.dll
│   │       ├── python3.dll
│   │       ├── python310._pth          # 경로 설정 파일
│   │       ├── DLLs/
│   │       └── Lib/
│   │           ├── site-packages/      # 모든 의존성 사전 설치
│   │           │   ├── pycaret/
│   │           │   ├── sklearn/
│   │           │   ├── shap/
│   │           │   ├── lime/
│   │           │   ├── fastapi/
│   │           │   ├── uvicorn/
│   │           │   ├── pandas/
│   │           │   ├── numpy/
│   │           │   └── ... (모든 ML 라이브러리)
│   │           └── ...
│   │
│   └── node_modules/                   # Electron 의존성
│
├── uploads/                            # 사용자 데이터 (실행 시 생성)
├── data/                               # 처리된 데이터 (실행 시 생성)
├── models/                             # 학습된 모델 (실행 시 생성)
│
└── electron-main.log                   # 로그 파일
```

---

## 빌드 프로세스

### Step 1: Python Runtime 준비

```batch
# 1. Python 3.10 Embedded 다운로드
# https://www.python.org/ftp/python/3.10.11/python-3.10.11-embed-amd64.zip

# 2. 압축 해제
cd resources
mkdir python-runtime
# python-3.10.11-embed-amd64.zip을 python-runtime/에 압축 해제

# 3. site-packages 활성화
# python310._pth 파일 수정:
#   - 'import site' 주석 제거
#   - Lib/site-packages 경로 추가

# 4. 개발 환경의 site-packages 복사
xcopy /E /I /Y backend\venv\Lib\site-packages python-runtime\Lib\site-packages
```

### Step 2: Frontend 빌드

```batch
cd frontend
npm run build
# 결과: frontend/build/
```

### Step 3: Electron 프로젝트 설정

```batch
# package.json 생성 (루트)
# electron-builder 설정
# main.ts 작성 (Electron 메인 프로세스)
```

### Step 4: Electron 빌드

```batch
npm run electron:compile  # TypeScript 컴파일
npm run electron:build:win  # Windows 빌드
```

---

## 실행 흐름

### 1. SambioWage.exe 시작
```
Electron Main Process 시작
  ↓
실행 폴더 확인 (getAppDataDir)
  ↓
uploads/, data/, models/ 폴더 생성
```

### 2. Backend 시작
```
python-runtime/python.exe 실행
  ↓
환경 변수 설정:
  - ELECTRON_APP=true
  - APP_DATA_DIR={실행폴더}
  - PORT=8000
  ↓
backend/run.py 실행
  ↓
FastAPI 서버 시작 (localhost:8000)
  ↓
Health Check 대기 (최대 30초)
```

### 3. Frontend 서빙
```
Express 서버 시작
  ↓
frontend/build/ 정적 파일 서빙 (localhost:3000)
  ↓
SPA 라우팅 설정 (/* → index.html)
```

### 4. BrowserWindow 생성
```
http://localhost:3000 로드
  ↓
React 앱 실행
  ↓
API 호출 → http://localhost:8000
```

---

## 예상 최종 크기

| 구성 요소 | 크기 |
|----------|------|
| Electron + Node.js | ~200MB |
| Python Runtime (embedded) | ~30MB |
| Python site-packages (ML 라이브러리) | ~600-800MB |
| Frontend (React build) | ~5MB |
| Backend (소스 코드) | ~5MB |
| **총 예상 크기** | **~850MB-1GB** |

압축 시: **~400-500MB**

---

## 장점 요약

### 개발자 관점
- ✅ 빌드 프로세스 단순화
- ✅ 디버깅 용이 (일반 Python 환경)
- ✅ 유지보수 편리
- ✅ 버그 수정 시 파일만 교체

### 사용자/고객사 관점
- ✅ 완전 오프라인 실행
- ✅ 인터넷 불필요
- ✅ 보안 정책 준수
- ✅ 설치 간편 (폴더 복사만)
- ✅ 이식성 (USB 등에서 실행 가능)

---

## 다음 단계

1. ✅ 전략 결정
2. ⏳ Python Embedded 다운로드 및 설정
3. ⏳ site-packages 준비
4. ⏳ Electron 프로젝트 초기화
5. ⏳ Frontend package.json 수정
6. ⏳ Electron main.ts 작성
7. ⏳ 빌드 및 테스트

---

**작성일**: 2025-10-27
**전략**: Python Embedded + 사전 설치 패키지
**예상 완료**: Phase 1-5 완료 후 Phase 2-5 진행

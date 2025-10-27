# SambioWage Electron 변환 프로젝트

## 프로젝트 개요

**목적**: 웹 기반 SambioWage (React + FastAPI)를 독립 실행 가능한 Electron 데스크톱 애플리케이션으로 변환

**참조 프로젝트**: `../SambioHRR` (Next.js + Electron + Python)

**선택 전략**: **전략 B - React + FastAPI 독립 유지**

---

## 목차

1. [현재 프로젝트 구조 분석](#1-현재-프로젝트-구조-분석)
2. [목표 아키텍처](#2-목표-아키텍처)
3. [변환 전략 비교](#3-변환-전략-비교)
4. [단계별 실행 계획](#4-단계별-실행-계획)
5. [주요 과제 및 해결 방안](#5-주요-과제-및-해결-방안)
6. [체크리스트](#6-체크리스트)

---

## 1. 현재 프로젝트 구조 분석

### 1.1 Frontend (React + CRA)

```
frontend/
├── src/
│   ├── App.tsx                 # React Router 설정
│   ├── index.tsx               # 엔트리포인트
│   ├── pages/                  # 페이지 컴포넌트
│   │   ├── DataUpload.tsx
│   │   ├── Modeling.tsx
│   │   ├── Analysis.tsx
│   │   ├── Dashboard.tsx
│   │   ├── Effects.tsx
│   │   └── StrategicDashboard.tsx
│   ├── components/             # UI 컴포넌트
│   └── lib/                    # 유틸리티
├── public/
└── package.json
```

**기술 스택**:
- React 19.1.0
- React Router v7
- Radix UI + Tailwind CSS
- Chart.js, Recharts
- Create React App (react-scripts 5.0.1)

**빌드 결과**: `frontend/build/` (정적 HTML/JS/CSS)

### 1.2 Backend (FastAPI + Python)

```
backend/
├── app/
│   ├── main.py                 # FastAPI 앱
│   ├── api/routes/             # API 엔드포인트
│   │   ├── data.py
│   │   ├── modeling.py
│   │   ├── analysis.py
│   │   ├── dashboard.py
│   │   └── strategic.py
│   ├── services/               # 비즈니스 로직
│   │   ├── data_service.py
│   │   ├── modeling_service.py
│   │   ├── analysis_service.py
│   │   └── dashboard_service.py
│   └── core/
│       └── config.py
├── uploads/                    # 업로드 파일 저장
├── data/                       # 데이터 파일
├── requirements.txt
└── run.py                      # Uvicorn 서버 실행
```

**기술 스택**:
- FastAPI + Uvicorn
- PyCaret 3.2.0 (AutoML)
- scikit-learn 1.2.2
- SHAP, LIME (모델 해석)
- pandas, openpyxl

**실행 방식**: `python run.py` → localhost:8000

### 1.3 현재 통신 구조

```
┌─────────────┐         HTTP API         ┌─────────────┐
│   React     │ ◄──────────────────────► │   FastAPI   │
│ (Port 3000) │         CORS             │ (Port 8000) │
└─────────────┘                          └─────────────┘
     Vercel                                   Railway
```

---

## 2. 목표 아키텍처

### 2.1 Electron 앱 구조

```
SambioWage.exe (Electron App)
│
├── [Main Process] Electron
│   ├── 프로세스 관리
│   ├── 창 관리
│   └── 리소스 조정
│
├── [Backend Process] Python FastAPI
│   ├── backend-server.exe (PyInstaller 번들)
│   ├── localhost:8000
│   └── 모든 ML 라이브러리 포함
│
└── [Frontend] React Static Files
    ├── Express로 서빙 (localhost:3000)
    └── build/ 폴더
```

### 2.2 파일 구조

```
release/win-unpacked/
├── SambioWage.exe              # Electron 실행 파일
├── resources/
│   ├── app/
│   │   ├── dist-electron/      # Electron 메인 프로세스
│   │   │   ├── main.js
│   │   │   └── preload.js
│   │   ├── frontend/build/     # React 빌드 결과
│   │   └── node_modules/
│   └── tools/
│       └── backend/
│           ├── backend-server.exe   # FastAPI 서버
│           └── _internal/           # Python 의존성
├── uploads/                    # 사용자 데이터 (실행 시 생성)
├── data/                       # 모델 파일 (실행 시 생성)
├── models/                     # 학습된 모델 (실행 시 생성)
└── electron-main.log           # 로그 파일
```

### 2.3 실행 흐름

```
1. SambioWage.exe 시작
   ↓
2. Electron Main Process 초기화
   ↓
3. Backend Process 시작 (backend-server.exe)
   ↓ (포트 8000 대기)
4. Backend Health Check
   ↓ (Ready!)
5. Frontend 서빙 시작 (Express, 포트 3000)
   ↓
6. BrowserWindow 생성 → http://localhost:3000 로드
   ↓
7. React App 로드 → API 호출 (http://localhost:8000)
   ↓
8. 사용자 인터페이스 표시
```

---

## 3. 변환 전략 비교

### 전략 A: Next.js 전환 (SambioHRR 패턴)
- ❌ React → Next.js 전환 작업량 막대
- ❌ FastAPI를 Next.js API Routes로 전환 불가능 (Python ML 라이브러리)
- ❌ 비현실적

### 전략 B: React + FastAPI 독립 유지 ✅ (채택)
- ✅ 기존 코드 최대한 재사용
- ✅ Frontend/Backend 독립성 유지
- ✅ SambioHRR의 Python 번들링 패턴 참조 가능
- ✅ 오프라인 독립 실행 가능
- ⚠️ 번들 크기 큼 (예상 800MB-1.5GB)
- ⚠️ 프로세스 관리 복잡도

### 전략 C: Backend 분리 배포
- ✅ 번들 크기 작음 (100MB 이하)
- ❌ 사용자 설정 복잡
- ❌ 오프라인 사용 불가
- ❌ 거부

---

## 4. 단계별 실행 계획

### Phase 1: Python Backend 독립 실행 파일화 (예상 1-2일)

#### 목표
FastAPI 서버를 PyInstaller로 단일 실행 파일로 번들링

#### 작업 항목

**1.1 PyInstaller 설정 파일 작성**
- [ ] `backend/backend-server.spec` 파일 생성
- [ ] 모든 PyCaret 의존성 포함 설정
- [ ] SHAP, LIME 모듈 포함 설정
- [ ] 숨겨진 imports 설정

**1.2 데이터 디렉토리 경로 수정**
- [ ] `backend/app/core/config.py` 수정
  - [ ] `get_data_dir()` 함수 추가 (실행 폴더 기준)
  - [ ] ELECTRON_APP 환경 변수 확인 로직
  - [ ] 동적 경로 설정 (uploads, data, models)
- [ ] 모든 파일 I/O 코드 확인 및 수정

**1.3 빌드 스크립트 작성**
- [ ] `backend/build-backend.bat` 작성
  - [ ] 가상환경 생성
  - [ ] 의존성 설치
  - [ ] PyInstaller 실행
  - [ ] 빌드 검증

**1.4 독립 실행 테스트**
- [ ] `backend/dist/backend-server/backend-server.exe` 실행
- [ ] http://localhost:8000 접속 확인
- [ ] /health 엔드포인트 확인
- [ ] API 엔드포인트 테스트 (Postman/curl)
- [ ] 파일 업로드 테스트
- [ ] 모델 학습 테스트

**예상 이슈**:
- PyCaret 의존성 누락 → `--collect-all pycaret` 추가
- SHAP native 라이브러리 누락 → `--collect-binaries shap` 추가
- 번들 크기 과도 (>1.5GB) → 불필요한 모듈 제외

---

### Phase 2: React Frontend Electron 준비 (예상 1일)

#### 목표
React 앱을 Electron 환경에서 실행 가능하도록 설정

#### 작업 항목

**2.1 package.json 설정**
- [ ] 루트 `package.json` 생성/수정
  - [ ] electron 의존성 추가
  - [ ] electron-builder 추가
  - [ ] 빌드 스크립트 추가
  - [ ] main 필드 설정

**2.2 Frontend 빌드 설정**
- [ ] `frontend/package.json` 수정
  - [ ] `"homepage": "."` 추가 (상대 경로)
  - [ ] `build:electron` 스크립트 추가
- [ ] 환경 변수 설정 방식 변경
  - [ ] 런타임 config.js 방식 검토
  - [ ] 또는 고정 포트 사용 (8000, 3000)

**2.3 빌드 테스트**
- [ ] `cd frontend && npm run build` 실행
- [ ] `build/` 폴더 생성 확인
- [ ] `build/index.html` 확인
- [ ] 상대 경로 확인 (/, /static/)

**2.4 로컬 서빙 테스트**
- [ ] Express로 `frontend/build` 서빙 테스트
- [ ] http://localhost:3000 접속 확인
- [ ] API 호출 확인 (http://localhost:8000)

---

### Phase 3: Electron Main Process 구현 (예상 2-3일)

#### 목표
Electron 메인 프로세스 작성 및 Backend/Frontend 통합

#### 작업 항목

**3.1 프로젝트 구조 설정**
- [ ] `electron/` 디렉토리 생성
- [ ] `electron/main.ts` 작성
- [ ] `electron/preload.ts` 작성
- [ ] `electron/tsconfig.json` 작성

**3.2 Backend 프로세스 관리 구현**
- [ ] `startBackend()` 함수 작성
  - [ ] backend-server.exe 경로 해석
  - [ ] spawn으로 프로세스 시작
  - [ ] 환경 변수 전달 (PORT, ELECTRON_APP, APP_DATA_DIR)
  - [ ] stdout/stderr 로깅
  - [ ] 에러 핸들링
- [ ] `waitForBackend()` 함수 작성
  - [ ] Health check 루프 (최대 30초)
  - [ ] 타임아웃 처리

**3.3 Frontend 서빙 구현**
- [ ] Express 서버 설정
  - [ ] `frontend/build` 정적 파일 서빙
  - [ ] SPA 라우팅 지원 (/* → index.html)
  - [ ] 포트 3000 설정
- [ ] 또는 `loadFile()` 방식 검토

**3.4 BrowserWindow 관리**
- [ ] `createWindow()` 함수 작성
  - [ ] 창 크기/설정 (1400x900)
  - [ ] webPreferences 설정
  - [ ] preload 스크립트 연결
- [ ] 이벤트 핸들러
  - [ ] ready-to-show
  - [ ] did-finish-load
  - [ ] did-fail-load

**3.5 프로세스 생명주기 관리**
- [ ] 앱 종료 시 Backend 프로세스 종료
- [ ] 비정상 종료 처리
- [ ] 재시작 로직 (선택사항)

**3.6 로깅 시스템**
- [ ] 실행 폴더에 `electron-main.log` 생성
- [ ] Backend 로그 파일 생성
- [ ] 타임스탬프 포함 로깅

**3.7 TypeScript 컴파일**
- [ ] `npm run electron:compile` 스크립트 작성
- [ ] `dist-electron/` 폴더에 컴파일 결과 생성

**3.8 개발 모드 테스트**
- [ ] Backend 수동 실행
- [ ] Frontend 개발 서버 실행
- [ ] `npm run electron:dev` 실행
- [ ] 통합 테스트

---

### Phase 4: 빌드 및 패키징 (예상 2일)

#### 목표
electron-builder로 배포 가능한 실행 파일 생성

#### 작업 항목

**4.1 electron-builder 설정**
- [ ] `package.json`의 `build` 섹션 작성
  - [ ] appId 설정
  - [ ] productName 설정
  - [ ] files 설정 (포함 파일)
  - [ ] extraResources 설정 (backend-server)
  - [ ] win 타겟 설정 (portable)

**4.2 아이콘 및 리소스**
- [ ] `public/icon.ico` 생성 (256x256)
- [ ] `public/icon.icns` 생성 (macOS, 선택사항)
- [ ] `public/icon.png` 생성 (Linux, 선택사항)

**4.3 빌드 스크립트 작성**
- [ ] `build-all.bat` 작성
  ```batch
  # 1. Backend 빌드
  cd backend
  call build-backend.bat
  cd ..

  # 2. Frontend 빌드
  cd frontend
  call npm run build
  cd ..

  # 3. Electron 컴파일
  npm run electron:compile

  # 4. Electron 빌드
  npm run electron:build:win
  ```

**4.4 첫 빌드 실행**
- [ ] `npm run electron:build:win` 실행
- [ ] 빌드 에러 수정
- [ ] `release/win-unpacked/` 확인

**4.5 실행 파일 테스트**
- [ ] `release/win-unpacked/SambioWage.exe` 실행
- [ ] Backend 프로세스 시작 확인
- [ ] Frontend 로딩 확인
- [ ] 전체 워크플로우 테스트
  - [ ] 데이터 업로드
  - [ ] 모델 학습
  - [ ] 분석
  - [ ] 대시보드

**4.6 다른 경로에서 테스트**
- [ ] `release/win-unpacked/` 폴더를 `D:\TestSambioWage\`로 복사
- [ ] 실행 및 동작 확인
- [ ] 경로 독립성 검증

---

### Phase 5: 최적화 및 테스트 (예상 2-3일)

#### 목표
성능 최적화, 안정성 강화, 사용자 경험 개선

#### 작업 항목

**5.1 초기 로딩 최적화**
- [ ] Splash screen 추가 (선택사항)
- [ ] 로딩 상태 표시
- [ ] Backend 시작 시간 측정 및 개선

**5.2 번들 크기 최적화**
- [ ] PyInstaller excludes 설정
  - [ ] 불필요한 PyCaret 모듈 제외
  - [ ] 테스트 파일 제외
  - [ ] 문서 파일 제외
- [ ] 최종 크기 목표: <1GB

**5.3 에러 처리 강화**
- [ ] Backend 시작 실패 시 명확한 에러 메시지
- [ ] 포트 충돌 감지 및 안내
- [ ] 로그 파일 위치 안내

**5.4 포트 충돌 방지**
- [ ] 동적 포트 할당 검토
- [ ] 또는 충돌 감지 후 재시도

**5.5 전체 시나리오 테스트**
- [ ] 첫 실행 (uploads, data 폴더 생성)
- [ ] 데이터 업로드
- [ ] 모델 학습 (Ridge, Lasso, GBR 등)
- [ ] SHAP/LIME 분석
- [ ] 대시보드 예측
- [ ] 앱 종료 및 재시작 (데이터 유지 확인)

**5.6 스트레스 테스트**
- [ ] 대용량 데이터 업로드 (1000+ 행)
- [ ] 여러 모델 동시 학습
- [ ] 메모리 누수 확인
- [ ] 장시간 실행 안정성

**5.7 문서 작성**
- [ ] `ELECTRON_BUILD_GUIDE.md` 작성 (이 문서 기반)
- [ ] 사용자 매뉴얼 작성
- [ ] 트러블슈팅 가이드

**5.8 배포 준비**
- [ ] 최종 빌드 생성
- [ ] 압축 파일 생성 (SambioWage-v1.0.0.zip)
- [ ] README 포함

---

## 5. 주요 과제 및 해결 방안

### 5.1 Python 번들 크기

**문제**: PyCaret + 모든 ML 라이브러리 = 500MB-1.2GB

**해결**:
```python
# backend-server.spec
excludes = [
    'matplotlib',  # 사용하지 않으면 제외
    'IPython',
    'notebook',
    'pytest',
    'sphinx',
]

a = Analysis(
    ['run.py'],
    excludes=excludes,
    ...
)
```

### 5.2 초기 로딩 시간

**문제**: FastAPI + PyCaret 로딩에 5-10초 소요

**해결**:
```typescript
// electron/main.ts
async function waitForBackend() {
  const startTime = Date.now();
  log('Starting backend server...');

  for (let i = 0; i < 30; i++) {
    try {
      const res = await fetch('http://localhost:8000/health');
      if (res.ok) {
        const elapsed = ((Date.now() - startTime) / 1000).toFixed(1);
        log(`✓ Backend ready in ${elapsed}s`);
        return true;
      }
    } catch {}
    await new Promise(r => setTimeout(r, 1000));
  }
  return false;
}
```

### 5.3 포트 충돌

**문제**: 8000, 3000 포트가 이미 사용 중

**해결 방안 1** (간단): 고정 포트 + 충돌 감지
```typescript
// 포트 사용 중이면 에러 메시지 표시
if (!await checkPort(8000)) {
  dialog.showErrorBox('Port Error',
    'Port 8000 is already in use. Please close other applications.');
  app.quit();
}
```

**해결 방안 2** (복잡): 동적 포트 할당
```typescript
import getPort from 'get-port';

const backendPort = await getPort({ port: 8000 });
const frontendPort = await getPort({ port: 3000 });

// React에 동적 API URL 전달
app.get('/config.js', (req, res) => {
  res.send(`window.REACT_APP_API_URL = 'http://localhost:${backendPort}';`);
});
```

### 5.4 데이터 경로 이식성

**문제**: 하드코딩된 경로는 다른 위치에서 작동 안 함

**해결**:
```python
# backend/app/core/config.py
import os
from pathlib import Path

def get_app_data_dir() -> Path:
    """실행 파일 위치 기준 데이터 디렉토리"""
    if os.getenv('ELECTRON_APP') == 'true':
        # Electron: APP_DATA_DIR 환경 변수 사용
        app_dir = Path(os.getenv('APP_DATA_DIR', '.'))
    else:
        # 개발 모드: 프로젝트 루트
        app_dir = Path(__file__).parent.parent.parent

    return app_dir.absolute()

# 모든 경로를 동적으로 설정
BASE_DIR = get_app_data_dir()
UPLOAD_DIR = BASE_DIR / 'uploads'
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# 디렉토리 생성
for dir_path in [UPLOAD_DIR, DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
```

### 5.5 PyInstaller 의존성 누락

**문제**: PyCaret의 숨겨진 의존성이 포함 안 됨

**해결**:
```python
# backend-server.spec
hiddenimports = [
    'pycaret.regression',
    'sklearn.utils._typedefs',
    'sklearn.utils._heap',
    'sklearn.utils._sorting',
    'sklearn.utils._vector_sentinel',
    'shap',
    'lime',
    'lime.lime_tabular',
]

a = Analysis(
    ['run.py'],
    hiddenimports=hiddenimports,
    ...
)
```

**검증**:
```batch
# 빌드 후 테스트
cd backend\dist\backend-server
backend-server.exe
# 에러 로그 확인 → 누락된 모듈 추가
```

### 5.6 Windows Defender 오탐

**문제**: PyInstaller 실행 파일이 바이러스로 오탐

**해결**:
1. 코드 서명 (권장, 비용 발생)
2. VirusTotal 스캔 후 화이트리스트 등록
3. 사용자에게 안내 (README)

---

## 6. 체크리스트

### 개발 환경 준비
- [ ] Node.js 18+ 설치 확인
- [ ] Python 3.11+ 설치 확인
- [ ] Git 설치 확인

### Phase 1: Backend 번들링
- [ ] PyInstaller spec 파일 작성
- [ ] 데이터 경로 수정
- [ ] 빌드 스크립트 작성
- [ ] 독립 실행 테스트 성공

### Phase 2: Frontend 준비
- [ ] package.json 설정
- [ ] 빌드 설정 변경
- [ ] 빌드 테스트 성공
- [ ] 로컬 서빙 테스트

### Phase 3: Electron 구현
- [ ] 프로젝트 구조 생성
- [ ] Backend 프로세스 관리 구현
- [ ] Frontend 서빙 구현
- [ ] BrowserWindow 생성
- [ ] 생명주기 관리
- [ ] 로깅 시스템
- [ ] TypeScript 컴파일
- [ ] 개발 모드 테스트

### Phase 4: 빌드 및 패키징
- [ ] electron-builder 설정
- [ ] 아이콘 생성
- [ ] 빌드 스크립트 작성
- [ ] 첫 빌드 성공
- [ ] 실행 파일 테스트
- [ ] 경로 독립성 검증

### Phase 5: 최적화 및 테스트
- [ ] 로딩 최적화
- [ ] 번들 크기 최적화
- [ ] 에러 처리 강화
- [ ] 전체 시나리오 테스트
- [ ] 스트레스 테스트
- [ ] 문서 작성
- [ ] 배포 준비

---

## 7. 예상 타임라인

| Phase | 작업 | 예상 시간 | 누적 시간 |
|-------|------|-----------|-----------|
| 1 | Backend 번들링 | 1-2일 | 1-2일 |
| 2 | Frontend 준비 | 1일 | 2-3일 |
| 3 | Electron 구현 | 2-3일 | 4-6일 |
| 4 | 빌드 및 패키징 | 2일 | 6-8일 |
| 5 | 최적화 및 테스트 | 2-3일 | 8-11일 |

**총 예상 기간**: 8-11일 (실 작업 시간 기준)

---

## 8. 참고 자료

### SambioHRR 참조 파일
- `../SambioHRR/Electron_build_guide.md`
- `../SambioHRR/electron/main.ts`
- `../SambioHRR/package.json`
- `../SambioHRR/excel-upload-server/build-excel-uploader.bat`

### 공식 문서
- [Electron 공식 문서](https://www.electronjs.org/docs)
- [electron-builder 문서](https://www.electron.build/)
- [PyInstaller 문서](https://pyinstaller.org/en/stable/)
- [FastAPI 문서](https://fastapi.tiangolo.com/)

---

**작성일**: 2025-10-27
**작성자**: Claude
**버전**: 1.0

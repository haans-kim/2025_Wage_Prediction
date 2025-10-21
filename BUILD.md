# SambioWage 프로젝트 빌드 가이드

이 문서는 Windows 환경에서 SambioWage 프로젝트를 빌드하고 실행하는 방법을 설명합니다.

## 시스템 요구사항

### 필수 소프트웨어
- **Python 3.10.x** (3.10.8 이상 권장)
- **Node.js** (14.x 이상)
- **npm** (Node.js와 함께 설치됨)
- **Git** (선택사항)

### Python 3.10 설치 확인
Windows에서 Python 3.10이 설치되어 있는지 확인:

```bash
py -3.10 --version
```

출력 예시: `Python 3.10.8` 또는 `Python 3.10.11`

Python 3.10이 설치되어 있지 않다면 [python.org](https://www.python.org/downloads/)에서 다운로드하여 설치하세요.

---

## 백엔드 (FastAPI) 설치 및 빌드

### 1. 백엔드 디렉토리로 이동
```bash
cd backend
```

### 2. Python 3.10 가상환경 생성
Windows에서는 `py -3.10` 런처를 사용하여 특정 Python 버전의 가상환경을 생성합니다:

```bash
py -3.10 -m venv venv
```

### 3. 가상환경 활성화

**PowerShell:**
```powershell
.\venv\Scripts\Activate.ps1
```

**명령 프롬프트(CMD):**
```cmd
venv\Scripts\activate.bat
```

**Git Bash:**
```bash
source venv/Scripts/activate
```

가상환경이 활성화되면 프롬프트 앞에 `(venv)`가 표시됩니다.

### 4. pip 업그레이드
```bash
python -m pip install --upgrade pip
```

### 5. 의존성 패키지 설치
```bash
pip install -r requirements.txt
```

이 과정은 다음 주요 패키지들을 설치합니다:
- **PyCaret 3.2.0** - AutoML 프레임워크
- **FastAPI** - 백엔드 API 프레임워크
- **scikit-learn 1.2.2** - 머신러닝 라이브러리
- **SHAP, LIME** - 모델 설명 가능성 라이브러리
- **explainerdashboard** - 대시보드 생성 도구
- **LightGBM, XGBoost** - 그래디언트 부스팅 모델
- 기타 데이터 처리 및 분석 라이브러리

설치 시간은 약 3-5분 소요될 수 있습니다.

### 6. 백엔드 서버 실행
```bash
python run.py
```

서버가 성공적으로 시작되면 다음과 같은 메시지가 표시됩니다:
```
INFO:     Uvicorn running on http://localhost:8000 (Press CTRL+C to quit)
```

백엔드 API는 기본적으로 `http://localhost:8000`에서 실행됩니다.

### 7. API 문서 확인
브라우저에서 다음 URL로 접속하여 자동 생성된 API 문서를 확인할 수 있습니다:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

## 프론트엔드 (React) 설치 및 빌드

### 1. 프론트엔드 디렉토리로 이동
새 터미널 창을 열고:

```bash
cd frontend
```

### 2. npm 의존성 설치
```bash
npm install
```

이 과정은 다음을 포함한 약 1500개의 패키지를 설치합니다:
- **React** - UI 라이브러리
- **TypeScript** - 타입 안전성
- **Tailwind CSS** - 스타일링
- **React Router** - 라우팅
- **Recharts** - 차트 라이브러리
- 기타 개발 도구

설치 시간은 네트워크 속도에 따라 1-3분 소요될 수 있습니다.

### 3. 개발 서버 실행
```bash
npm start
```

개발 서버가 시작되면 자동으로 브라우저가 열립니다.
프론트엔드는 기본적으로 `http://localhost:3000`에서 실행됩니다.

### 4. 프로덕션 빌드 (배포용)
프로덕션 환경용 최적화된 빌드를 생성하려면:

```bash
npm run build
```

빌드된 파일은 `frontend/build/` 디렉토리에 생성됩니다.

---

## 전체 애플리케이션 실행

### 동시 실행 방법

1. **첫 번째 터미널 - 백엔드:**
   ```bash
   cd backend
   .\venv\Scripts\activate  # 가상환경 활성화
   python run.py
   ```

2. **두 번째 터미널 - 프론트엔드:**
   ```bash
   cd frontend
   npm start
   ```

3. **브라우저 접속:**
   - 프론트엔드: `http://localhost:3000`
   - 백엔드 API: `http://localhost:8000`
   - API 문서: `http://localhost:8000/docs`

---

## 환경 설정

### 백엔드 환경 변수 (선택사항)
백엔드 디렉토리에 `.env` 파일을 생성하여 설정을 커스터마이즈할 수 있습니다:

```env
HOST=0.0.0.0
PORT=8000
RELOAD=true
```

### 프론트엔드 환경 변수
프론트엔드 디렉토리에 `.env` 파일을 생성하여 백엔드 API URL을 설정할 수 있습니다:

```env
REACT_APP_API_URL=http://localhost:8000
```

---

## 문제 해결

### 백엔드 관련

**문제:** `py -3.10` 명령을 찾을 수 없음
- **해결:** Python 3.10을 설치하거나, `python --version`으로 현재 버전 확인 후 `python -m venv venv` 사용

**문제:** 가상환경 활성화 시 권한 오류 (PowerShell)
- **해결:** PowerShell을 관리자 권한으로 실행 후 다음 명령 실행:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

**문제:** pip 설치 중 빌드 오류
- **해결:** Microsoft C++ Build Tools 설치 필요
  - [Visual Studio Build Tools](https://visualstudio.microsoft.com/downloads/) 다운로드
  - "C++를 사용한 데스크톱 개발" 워크로드 선택

**문제:** 포트 8000이 이미 사용 중
- **해결:** `run.py`에서 포트 번호를 변경하거나, 다른 프로세스 종료

### 프론트엔드 관련

**문제:** npm 설치 중 네트워크 오류
- **해결:** npm 캐시 정리 후 재시도:
  ```bash
  npm cache clean --force
  npm install
  ```

**문제:** 포트 3000이 이미 사용 중
- **해결:** 프롬프트에서 다른 포트(예: 3001) 사용 여부 확인 시 `Y` 선택

**문제:** 백엔드 API 연결 실패
- **해결:**
  1. 백엔드 서버가 실행 중인지 확인 (`http://localhost:8000/docs` 접속 테스트)
  2. `.env` 파일의 `REACT_APP_API_URL` 확인
  3. CORS 설정 확인

---

## 테스트

### 백엔드 테스트
현재 자동화된 백엔드 테스트는 구현되어 있지 않습니다.
API 엔드포인트는 Swagger UI (`http://localhost:8000/docs`)에서 수동으로 테스트할 수 있습니다.

### 프론트엔드 테스트
```bash
cd frontend
npm test
```

---

## 추가 정보

### 프로젝트 구조
```
2025_Wage_Prediction/
├── backend/              # FastAPI 백엔드
│   ├── app/
│   │   ├── api/         # API 라우트
│   │   ├── services/    # 비즈니스 로직
│   │   └── main.py      # FastAPI 앱
│   ├── venv/            # Python 가상환경
│   ├── requirements.txt # Python 의존성
│   └── run.py           # 서버 실행 스크립트
├── frontend/            # React 프론트엔드
│   ├── src/
│   │   ├── pages/       # 페이지 컴포넌트
│   │   ├── components/  # 재사용 컴포넌트
│   │   └── lib/         # 유틸리티 (API 클라이언트 등)
│   ├── package.json     # npm 의존성
│   └── build/           # 프로덕션 빌드 (생성됨)
└── BUILD.md             # 이 문서
```

### 주요 기능
- 데이터 업로드 및 검증
- 자동 머신러닝 모델 학습 (PyCaret)
- 모델 설명 가능성 (SHAP, LIME)
- 임금 인상 예측 대시보드
- 시나리오 분석

### 관련 문서
- [CLAUDE.md](./CLAUDE.md) - 프로젝트 개요 및 개발 가이드
- [README.md](./README.md) - 프로젝트 소개

---

## 지원

문제가 발생하거나 질문이 있으면 프로젝트 관리자에게 문의하세요.

**마지막 업데이트:** 2025-10-21

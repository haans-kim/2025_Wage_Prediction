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
        # Electron 환경: APP_DATA_DIR 환경 변수 사용
        app_dir = Path(os.getenv('APP_DATA_DIR', '.'))
    else:
        # 개발 환경: 프로젝트 루트 (backend 상위 디렉토리)
        app_dir = Path(__file__).parent.parent.parent

    return app_dir.absolute()

# 기본 데이터 디렉토리
BASE_DIR = get_app_data_dir()
UPLOAD_DIR = BASE_DIR / 'uploads'
DATA_DIR = BASE_DIR / 'data'
MODELS_DIR = BASE_DIR / 'models'

# 디렉토리 생성
for dir_path in [UPLOAD_DIR, DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

class Settings(BaseSettings):
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "SambioWage"
    VERSION: str = "1.0.0"

    # CORS 설정
    ALLOWED_HOSTS: List[str] = [
        "http://localhost:3000",
        "http://localhost:3001",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:3001",
        "*"  # 개발 환경에서 모든 도메인 허용
    ]

    # 파일 업로드 설정
    MAX_FILE_SIZE: int = 10 * 1024 * 1024  # 10MB
    UPLOAD_DIR: str = str(UPLOAD_DIR)
    ALLOWED_FILE_TYPES: List[str] = [".xlsx", ".xls", ".csv"]

    # ML 모델 설정
    MODEL_DIR: str = str(MODELS_DIR)
    DATA_DIR: str = str(DATA_DIR)

    class Config:
        env_file = ".env"

settings = Settings()
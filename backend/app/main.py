import os
import random
import numpy as np

# 재현성을 위한 환경 변수 및 시드 설정
os.environ['PYTHONHASHSEED'] = '42'
random.seed(42)
np.random.seed(42)

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from app.api.routes import data, modeling, analysis, dashboard
from app.core.config import settings

app = FastAPI(
    title="SambioWage API",
    description="머신러닝 기반 임금인상률 예측 대시보드 API",
    version="1.0.0"
)

# 수동 CORS 헤더 추가
@app.middleware("http")
async def cors_handler(request: Request, call_next):
    response = await call_next(request)
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 라우터 등록
app.include_router(data.router, prefix="/api/data", tags=["data"])
app.include_router(modeling.router, prefix="/api/modeling", tags=["modeling"])
app.include_router(analysis.router, prefix="/api/analysis", tags=["analysis"])
app.include_router(dashboard.router, prefix="/api/dashboard", tags=["dashboard"])

@app.get("/")
async def root():
    return {"message": "SambioWage API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
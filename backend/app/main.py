from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi import Request
from app.api.routes import data, modeling, analysis, dashboard, strategic
from app.core.config import settings
import os

app = FastAPI(
    title="SambioWage API",
    description="머신러닝 기반 임금인상률 예측 대시보드 API",
    version="1.0.0"
)

# CORS 설정 - 환경에 따라 다르게 적용
if os.getenv("ENVIRONMENT") == "production":
    # Production CORS 설정
    origins = os.getenv("CORS_ORIGINS", "").split(",")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins[0] else ["https://sambiowage.vercel.app"],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
else:
    # Development CORS 설정 (모든 origin 허용)
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
app.include_router(strategic.router, prefix="/api/strategic", tags=["strategic"])

@app.get("/")
async def root():
    return {"message": "SambioWage API", "version": "1.0.0"}

@app.get("/health")
async def health_check():
    return {"status": "healthy"}
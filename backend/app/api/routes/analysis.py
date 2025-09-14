from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import pandas as pd
import numpy as np
from app.services.modeling_service import modeling_service
from app.services.analysis_service import analysis_service

router = APIRouter()

class FeatureAnalysisRequest(BaseModel):
    sample_index: Optional[int] = None
    top_n_features: int = 10

@router.get("/shap")
async def get_shap_analysis(
    sample_index: Optional[int] = Query(None),
    top_n: int = Query(10, ge=1, le=50)
) -> Dict[str, Any]:
    """
    SHAP 분석 결과 반환
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = analysis_service.get_shap_analysis(
            model=modeling_service.current_model,
            sample_index=sample_index,
            top_n=top_n
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"SHAP analysis failed: {str(e)}")

@router.get("/feature-importance")
async def get_feature_importance(
    method: str = Query("shap", regex="^(shap|permutation|built_in)$"),
    top_n: int = Query(15, ge=1, le=50)
) -> Dict[str, Any]:
    """
    Feature importance 분석
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = analysis_service.get_feature_importance(
            model=modeling_service.current_model,
            method=method,
            top_n=top_n
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance analysis failed: {str(e)}")

@router.get("/lime")
async def get_lime_analysis(
    sample_index: int = Query(..., ge=0),
    num_features: int = Query(10, ge=1, le=20)
) -> Dict[str, Any]:
    """
    LIME 분석 결과 반환 (개별 예측 설명)
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = analysis_service.get_lime_analysis(
            model=modeling_service.current_model,
            sample_index=sample_index,
            num_features=num_features
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LIME analysis failed: {str(e)}")

@router.get("/model-performance")
async def get_model_performance() -> Dict[str, Any]:
    """
    모델 성능 메트릭 상세 분석
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = analysis_service.get_model_performance_analysis()
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Performance analysis failed: {str(e)}")

@router.get("/partial-dependence")
async def get_partial_dependence(
    feature_name: str = Query(...),
    num_grid_points: int = Query(50, ge=10, le=200)
) -> Dict[str, Any]:
    """
    부분 의존성 플롯 (Partial Dependence Plot) 데이터
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = analysis_service.get_partial_dependence(
            model=modeling_service.current_model,
            feature_name=feature_name,
            num_grid_points=num_grid_points
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Partial dependence analysis failed: {str(e)}")

@router.get("/residual-analysis")
async def get_residual_analysis() -> Dict[str, Any]:
    """
    잔차 분석 (Residual Analysis)
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = analysis_service.get_residual_analysis(modeling_service.current_model)
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Residual analysis failed: {str(e)}")

@router.get("/prediction-intervals")
async def get_prediction_intervals(
    confidence_level: float = Query(0.95, ge=0.8, le=0.99)
) -> Dict[str, Any]:
    """
    예측 구간 (Prediction Intervals) 분석
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = analysis_service.get_prediction_intervals(
            model=modeling_service.current_model,
            confidence_level=confidence_level
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction intervals analysis failed: {str(e)}")

# 레거시 엔드포인트 (호환성을 위해 유지)
@router.get("/explainer")
async def get_explainer_dashboard() -> Dict[str, Any]:
    """
    ExplainerDashboard 종합 데이터 반환
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        # 여러 분석 결과를 종합하여 반환
        shap_result = analysis_service.get_shap_analysis(modeling_service.current_model)
        feature_importance = analysis_service.get_feature_importance(modeling_service.current_model)
        performance = analysis_service.get_model_performance_analysis()
        
        return {
            "message": "Explainer dashboard data loaded successfully",
            "shap_analysis": shap_result,
            "feature_importance": feature_importance,
            "model_performance": performance
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Explainer analysis failed: {str(e)}")


@router.post("/clear-cache")
async def clear_analysis_cache() -> Dict[str, Any]:
    """
    분석 캐시 초기화 (SHAP, Feature Importance 등)
    """
    try:
        # analysis_service 캐시 초기화
        analysis_service._shap_cache = {}
        analysis_service._importance_cache = {}
        analysis_service._last_model_id = None
        
        return {
            "message": "Analysis cache cleared successfully",
            "cleared_caches": ["shap_cache", "importance_cache"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear cache: {str(e)}")
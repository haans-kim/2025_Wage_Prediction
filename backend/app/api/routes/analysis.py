from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
import pandas as pd
import numpy as np
from app.services.modeling_service import modeling_service
from app.services.analysis_service import analysis_service
from app.services.explainer_dashboard_service import explainer_dashboard_service

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

@router.get("/explainer-dashboard")
async def get_explainer_dashboard_status() -> Dict[str, Any]:
    """
    ExplainerDashboard 상태 확인
    """
    try:
        status = explainer_dashboard_service.get_status()
        return status
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get dashboard status: {str(e)}")

@router.post("/explainer-dashboard")
async def create_explainer_dashboard() -> Dict[str, Any]:
    """
    ExplainerDashboard 생성 및 실행
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        # PyCaret 환경에 의존하지 않고 데이터 직접 생성
        try:
            # 먼저 PyCaret 데이터 시도
            X_train, y_train, X_test, y_test = analysis_service._get_training_data()
        except Exception as e:
            print(f"PyCaret data failed, using fallback: {e}")
            # Fallback: data_service에서 직접 데이터 생성
            from app.services.data_service import data_service
            if data_service.current_data is not None:
                data = data_service.current_data.copy()
                
                # modeling_service에서 모델이 훈련된 피처 정보 가져오기
                if hasattr(modeling_service, 'feature_names') and modeling_service.feature_names:
                    model_features = modeling_service.feature_names
                    print(f"Model was trained with features: {model_features[:5]}...")
                    
                    # 데이터에서 모델 피처만 선택
                    available_features = [f for f in model_features if f in data.columns]
                    print(f"Available model features in data: {len(available_features)}/{len(model_features)}")
                    
                    if available_features:
                        X_test = data[available_features].head(10)  # 상위 10개 행만 사용
                        y_test = pd.Series([0.05] * len(X_test))  # 5% 기본 인상률
                        print(f"Created X_test with model features: {X_test.shape}")
                    else:
                        raise HTTPException(status_code=404, detail="No matching features found between model and data")
                else:
                    # 피처 정보가 없으면 수치형 데이터 사용
                    print("No model feature info, using numeric columns")
                    X_test = data.select_dtypes(include=[np.number]).head(10)
                    y_test = pd.Series([0.05] * len(X_test))
                    print(f"Created X_test: {X_test.shape}, y_test: {len(y_test)}")
            else:
                raise HTTPException(status_code=404, detail="No data available for dashboard creation")
        
        # Feature names 가져오기
        feature_names = list(X_test.columns) if hasattr(X_test, 'columns') else None
        if not feature_names:
            # modeling_service에서 feature names 가져오기
            if hasattr(modeling_service, 'feature_names') and modeling_service.feature_names:
                feature_names = modeling_service.feature_names
            else:
                feature_names = [f"feature_{i}" for i in range(X_test.shape[1])]
        
        print(f"Dashboard creation: X_test.shape={X_test.shape}, feature_names={len(feature_names)}")
        
        # 대시보드 생성
        result = explainer_dashboard_service.create_dashboard(
            model=modeling_service.current_model,
            X_test=X_test,
            y_test=y_test,
            feature_names=feature_names,
            model_name="Wage Increase Prediction Model"
        )
        
        if result["success"]:
            return result
        else:
            raise HTTPException(status_code=500, detail=result["error"])
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create ExplainerDashboard: {str(e)}")

@router.delete("/explainer-dashboard")
async def stop_explainer_dashboard() -> Dict[str, Any]:
    """
    ExplainerDashboard 중지
    """
    try:
        explainer_dashboard_service.stop_dashboard()
        return {"message": "ExplainerDashboard stopped successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to stop dashboard: {str(e)}")

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
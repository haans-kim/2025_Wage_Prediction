from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel
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
    top_n: int = Query(10, ge=1, le=50),
    target: str = Query("wage_increase_bu_sbl", description="Target to analyze: wage_increase_bu_sbl or wage_increase_mi_sbl")
) -> Dict[str, Any]:
    """
    SHAP 분석 결과 반환 (Base-up 또는 성과급)
    """
    try:
        # Target에 따른 모델 선택
        if target == "wage_increase_bu_sbl":
            model = modeling_service.baseup_model if hasattr(modeling_service, 'baseup_model') else modeling_service.current_model
        elif target == "wage_increase_mi_sbl":
            model = modeling_service.performance_model if hasattr(modeling_service, 'performance_model') else modeling_service.current_model
        else:
            model = modeling_service.current_model
            
        if not model:
            raise HTTPException(status_code=404, detail=f"No trained model available for {target}")
        
        result = analysis_service.get_shap_analysis(
            model=model,
            sample_index=sample_index,
            top_n=top_n
        )
        # Target 정보를 결과에 추가
        result['target'] = target
        
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
        
        # 훈련 데이터 가져오기
        X_train, y_train, X_test, y_test = analysis_service._get_training_data()
        
        if X_test is None or y_test is None:
            raise HTTPException(status_code=404, detail="No test data available")
        
        # Feature names 가져오기
        feature_names = analysis_service.feature_names
        
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

class WhatIfScenario(BaseModel):
    scenario_type: str  # recession, inflation, boom, custom
    gdp_change: Optional[float] = None
    cpi_change: Optional[float] = None
    major_group_rate_change: Optional[float] = None
    revenue_growth_change: Optional[float] = None
    custom_features: Optional[Dict[str, float]] = None

@router.post("/what-if-scenario")
async def analyze_what_if_scenario(scenario: WhatIfScenario) -> Dict[str, Any]:
    """
    What-If 시나리오 분석 - 경제 시나리오별 임금인상률 예측
    """
    try:
        # 시나리오별 기본값 설정
        scenario_presets = {
            "recession": {"gdp_change": -2.0, "cpi_change": -1.0, "major_group_rate_change": -1.0},
            "inflation": {"gdp_change": 0.5, "cpi_change": 5.0, "major_group_rate_change": 2.0},
            "boom": {"gdp_change": 4.0, "cpi_change": 3.0, "major_group_rate_change": 3.0, "revenue_growth_change": 20.0},
            "custom": {}
        }
        
        # 시나리오 적용
        changes = scenario_presets.get(scenario.scenario_type, {})
        if scenario.gdp_change is not None:
            changes["gdp_change"] = scenario.gdp_change
        if scenario.cpi_change is not None:
            changes["cpi_change"] = scenario.cpi_change
        if scenario.major_group_rate_change is not None:
            changes["major_group_rate_change"] = scenario.major_group_rate_change
        if scenario.revenue_growth_change is not None:
            changes["revenue_growth_change"] = scenario.revenue_growth_change
        if scenario.custom_features:
            changes.update(scenario.custom_features)
        
        # 두 모델에 대한 예측
        results = {
            "scenario": scenario.scenario_type,
            "changes": changes,
            "predictions": {}
        }
        
        # Base-up 모델 예측
        if hasattr(modeling_service, 'baseup_model') and modeling_service.baseup_model:
            baseup_prediction = analysis_service.predict_with_scenario(
                modeling_service.baseup_model, 
                changes, 
                "baseup"
            )
            results["predictions"]["baseup"] = baseup_prediction
        
        # 성과급 모델 예측
        if hasattr(modeling_service, 'performance_model') and modeling_service.performance_model:
            performance_prediction = analysis_service.predict_with_scenario(
                modeling_service.performance_model, 
                changes, 
                "performance"
            )
            results["predictions"]["performance"] = performance_prediction
        
        # 총 인상률 계산
        if "baseup" in results["predictions"] and "performance" in results["predictions"]:
            results["predictions"]["total"] = {
                "current": results["predictions"]["baseup"]["current"] + results["predictions"]["performance"]["current"],
                "scenario": results["predictions"]["baseup"]["scenario"] + results["predictions"]["performance"]["scenario"],
                "change": results["predictions"]["baseup"]["change"] + results["predictions"]["performance"]["change"]
            }
        
        return results
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"What-If scenario analysis failed: {str(e)}")
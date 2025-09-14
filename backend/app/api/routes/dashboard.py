from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, List, Optional
from pydantic import BaseModel
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from app.services.modeling_service import modeling_service
from app.services.dashboard_service import dashboard_service

router = APIRouter()

class ScenarioVariable(BaseModel):
    name: str
    value: float
    min_value: float
    max_value: float
    description: str

class ScenarioRequest(BaseModel):
    scenario_name: str
    variables: Dict[str, float]
    description: Optional[str] = None

class PredictionRequest(BaseModel):
    input_data: Dict[str, float]
    confidence_level: float = 0.95

@router.post("/predict")
async def predict_wage_increase(request: PredictionRequest) -> Dict[str, Any]:
    """
    2025년 임금인상률 예측
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(
                status_code=404, 
                detail={
                    "error": "No trained model available",
                    "message": "모델이 훈련되지 않았습니다. Analysis 페이지에서 먼저 모델을 훈련해주세요.",
                    "redirect_to": "/analysis",
                    "steps": [
                        "1. Analysis 페이지로 이동",
                        "2. 환경 설정 버튼 클릭",  
                        "3. 모델 비교 수행",
                        "4. 모델 훈련 완료 후 Dashboard 재방문"
                    ]
                }
            )
        
        result = dashboard_service.predict_wage_increase(
            model=modeling_service.current_model,
            input_data=request.input_data,
            confidence_level=request.confidence_level
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@router.post("/scenario-analysis")
async def run_scenario_analysis(scenarios: List[ScenarioRequest]) -> Dict[str, Any]:
    """
    다중 시나리오 분석 실행
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = dashboard_service.run_scenario_analysis(
            model=modeling_service.current_model,
            scenarios=scenarios
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenario analysis failed: {str(e)}")

@router.get("/variables")
async def get_available_variables() -> Dict[str, Any]:
    """
    시나리오 분석에 사용 가능한 변수 목록 및 현재 값
    """
    try:
        result = dashboard_service.get_available_variables()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get variables: {str(e)}")

@router.get("/historical-trends")
async def get_historical_trends(
    years: int = Query(10, ge=1, le=20),
    include_forecast: bool = Query(True)
) -> Dict[str, Any]:
    """
    과거 임금인상률 트렌드 및 예측
    """
    try:
        result = dashboard_service.get_historical_trends(
            years=years,
            include_forecast=include_forecast
        )
        
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get historical trends: {str(e)}")

@router.get("/economic-indicators")
async def get_economic_indicators() -> Dict[str, Any]:
    """
    주요 경제 지표 현황
    """
    try:
        result = dashboard_service.get_economic_indicators()
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get economic indicators: {str(e)}")

@router.get("/scenario-templates")
async def get_scenario_templates() -> Dict[str, Any]:
    """
    사전 정의된 시나리오 템플릿
    """
    try:
        templates = dashboard_service.get_scenario_templates()
        return {
            "message": "Scenario templates retrieved successfully",
            "templates": templates
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get scenario templates: {str(e)}")

@router.post("/sensitivity-analysis")
async def run_sensitivity_analysis(
    base_scenario: Dict[str, float],
    variable_name: str,
    variation_range: float = 0.2
) -> Dict[str, Any]:
    """
    민감도 분석 - 특정 변수 변화에 따른 예측값 변화
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = dashboard_service.run_sensitivity_analysis(
            model=modeling_service.current_model,
            base_scenario=base_scenario,
            variable_name=variable_name,
            variation_range=variation_range
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensitivity analysis failed: {str(e)}")

@router.get("/forecast-accuracy")
async def get_forecast_accuracy() -> Dict[str, Any]:
    """
    예측 정확도 분석
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = dashboard_service.get_forecast_accuracy(modeling_service.current_model)
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get forecast accuracy: {str(e)}")

@router.post("/monte-carlo")
async def run_monte_carlo_simulation(
    base_scenario: Dict[str, float],
    uncertainty_ranges: Dict[str, float],
    num_simulations: int = Query(1000, ge=100, le=10000)
) -> Dict[str, Any]:
    """
    몬테카를로 시뮬레이션을 통한 불확실성 분석
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = dashboard_service.run_monte_carlo_simulation(
            model=modeling_service.current_model,
            base_scenario=base_scenario,
            uncertainty_ranges=uncertainty_ranges,
            num_simulations=num_simulations
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Monte Carlo simulation failed: {str(e)}")

@router.get("/market-conditions")
async def get_market_conditions() -> Dict[str, Any]:
    """
    현재 시장 상황 요약
    """
    try:
        result = dashboard_service.get_market_conditions()
        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get market conditions: {str(e)}")


@router.get("/model-feature-importance")
async def get_model_feature_importance() -> Dict[str, Any]:
    """
    실제 학습된 모델의 Feature Importance 반환
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail={"error": "No trained model available", "message": "모델을 먼저 학습시켜주세요"})

        # Analysis 서비스에서 Feature Importance 가져오기
        from app.services.analysis_service import analysis_service

        # Permutation Importance 사용 (더 안정적)
        importance_data = analysis_service.get_feature_importance(
            model=modeling_service.current_model,
            method='permutation',
            top_n=15
        )

        if importance_data and 'feature_importance' in importance_data:
            return importance_data
        else:
            # SHAP 시도
            importance_data = analysis_service.get_feature_importance(
                model=modeling_service.current_model,
                method='shap',
                top_n=15
            )
            if importance_data and 'feature_importance' in importance_data:
                return importance_data

        # 둘 다 실패하면 기본값
        return {
            "feature_importance": [],
            "message": "Feature importance calculation failed, using defaults"
        }

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get feature importance: {str(e)}")

@router.post("/custom-scenario")
async def create_custom_scenario(
    scenario_name: str,
    variables: Dict[str, float],
    save_template: bool = False
) -> Dict[str, Any]:
    """
    사용자 정의 시나리오 생성 및 예측
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        result = dashboard_service.create_custom_scenario(
            model=modeling_service.current_model,
            scenario_name=scenario_name,
            variables=variables,
            save_template=save_template
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Custom scenario creation failed: {str(e)}")

@router.get("/prediction-breakdown")
async def get_prediction_breakdown(
    scenario_variables: Optional[str] = Query(None, description="JSON string of scenario variables")
) -> Dict[str, Any]:
    """
    예측 결과의 상세 분해 (기여도 분석)
    """
    try:
        if not modeling_service.current_model:
            raise HTTPException(status_code=404, detail="No trained model available")
        
        import json
        variables = json.loads(scenario_variables) if scenario_variables else {}
        
        result = dashboard_service.get_prediction_breakdown(
            model=modeling_service.current_model,
            variables=variables
        )
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction breakdown failed: {str(e)}")

@router.get("/trend-data")
async def get_trend_data() -> Dict[str, Any]:
    """
    과거 임금인상률 추이 및 2025년 예측 데이터 (트렌드 차트용)
    """
    try:
        result = dashboard_service.get_trend_data()
        
        return result
        
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Trend data failed: {str(e)}")
"""
Strategic Dashboard and Hybrid Prediction API Routes
전략적 대시보드 및 하이브리드 예측 API
"""
from fastapi import APIRouter, HTTPException, Query
from typing import Dict, Any, Optional, List
from pydantic import BaseModel

from app.services.strategic_dashboard_service import strategic_dashboard_service
from app.services.hybrid_prediction_service import hybrid_prediction_service
from app.services.modeling_service import modeling_service


router = APIRouter()


class PredictionRequest(BaseModel):
    """예측 요청 모델"""
    year: int = 2026
    scenario: str = 'base'  # base, conservative, optimistic, custom
    custom_params: Optional[Dict[str, float]] = None


class SimulatorRequest(BaseModel):
    """시뮬레이터 요청 모델"""
    min_wage: float = 3.5
    gdp: float = 1.6
    revenue_growth: float = 15.0
    profit_margin: float = 12.0
    industry_avg: float = 4.8
    crisis_score: float = 0.3


@router.get("/dashboard")
async def get_strategic_dashboard() -> Dict[str, Any]:
    """
    전략적 대시보드 전체 데이터 조회
    """
    try:
        dashboard = strategic_dashboard_service.generate_dashboard()
        return {
            "message": "Strategic dashboard generated successfully",
            "dashboard": dashboard
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Dashboard generation error: {str(e)}")


@router.post("/predict")
async def predict_wage_increase(request: PredictionRequest) -> Dict[str, Any]:
    """
    하이브리드 방식으로 임금인상률 예측

    - 전략적 규칙 (70%)
    - ML 검증 (20%)
    - 잔차 학습 (10%)
    """
    try:
        result = hybrid_prediction_service.predict_wage_increase(
            year=request.year,
            scenario=request.scenario,
            custom_params=request.custom_params
        )

        return {
            "message": "Prediction completed successfully",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@router.post("/simulate")
async def simulate_scenario(request: SimulatorRequest) -> Dict[str, Any]:
    """
    실시간 시나리오 시뮬레이션
    """
    try:
        result = strategic_dashboard_service.create_interactive_simulator(
            min_wage=request.min_wage,
            gdp=request.gdp,
            revenue_growth=request.revenue_growth,
            profit_margin=request.profit_margin,
            industry_avg=request.industry_avg,
            crisis_score=request.crisis_score
        )

        return {
            "message": "Simulation completed successfully",
            "result": result
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Simulation error: {str(e)}")


@router.get("/scenarios")
async def get_scenarios() -> Dict[str, Any]:
    """
    사전 정의된 시나리오 목록 조회
    """
    try:
        scenarios = {}

        # 각 시나리오별 예측 실행
        for scenario_key, scenario_data in strategic_dashboard_service.scenarios.items():
            result = strategic_dashboard_service._calculate_scenario(scenario_key)
            scenarios[scenario_key] = {
                'name': scenario_data['name'],
                'description': scenario_data['description'],
                'parameters': scenario_data['params'],
                'prediction': {
                    'base_up': result['base_up'],
                    'mi': result['mi'],
                    'total': result['total']
                }
            }

        return {
            "message": "Scenarios retrieved successfully",
            "scenarios": scenarios
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenarios retrieval error: {str(e)}")


@router.get("/sensitivity")
async def get_sensitivity_analysis(
    scenario: str = Query('base', description="Scenario to analyze")
) -> Dict[str, Any]:
    """
    민감도 분석 수행
    """
    try:
        # 기본 시나리오 파라미터 가져오기
        scenario_map = {
            'base': '기본_시나리오',
            'conservative': '보수적_시나리오',
            'optimistic': '낙관적_시나리오'
        }
        scenario_key = scenario_map.get(scenario, '기본_시나리오')

        # 민감도 분석 실행
        sensitivity = strategic_dashboard_service._create_sensitivity_analysis()

        return {
            "message": "Sensitivity analysis completed",
            "analysis": sensitivity
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensitivity analysis error: {str(e)}")


@router.get("/historical-patterns")
async def get_historical_patterns() -> Dict[str, Any]:
    """
    역사적 패턴 분석 조회
    """
    try:
        patterns = strategic_dashboard_service._create_historical_analysis()

        return {
            "message": "Historical patterns retrieved successfully",
            "patterns": patterns
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Historical patterns error: {str(e)}")


@router.get("/feature-importance")
async def get_feature_importance() -> Dict[str, Any]:
    """
    Feature Importance 분석 (ML + Strategic Rules)
    """
    try:
        importance = hybrid_prediction_service.extract_feature_importance()

        return {
            "message": "Feature importance extracted successfully",
            "importance": importance
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")


@router.get("/decision-rules")
async def get_decision_rules() -> Dict[str, Any]:
    """
    의사결정 규칙 조회
    """
    try:
        rules = strategic_dashboard_service._create_decision_rules_panel()

        return {
            "message": "Decision rules retrieved successfully",
            "rules": rules
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Decision rules error: {str(e)}")


@router.get("/validation")
async def validate_predictions() -> Dict[str, Any]:
    """
    예측 품질 검증
    """
    try:
        validation = hybrid_prediction_service.validate_prediction_quality()

        return {
            "message": "Validation completed successfully",
            "validation": validation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Validation error: {str(e)}")


@router.post("/compare-models")
async def compare_prediction_models(scenario: str = 'base') -> Dict[str, Any]:
    """
    전략적 규칙 vs ML 모델 비교
    """
    try:
        # 전략적 예측
        strategic_pred = hybrid_prediction_service._get_strategic_prediction(scenario, None)

        # ML 예측
        ml_pred = hybrid_prediction_service._get_ml_validation(scenario, None)

        # 하이브리드 예측
        hybrid_pred = hybrid_prediction_service.predict_wage_increase(
            year=2026,
            scenario=scenario
        )

        comparison = {
            'strategic': strategic_pred,
            'ml': ml_pred if ml_pred else {'message': 'No ML model available'},
            'hybrid': hybrid_pred['prediction'],
            'differences': {
                'strategic_vs_ml': (
                    round(strategic_pred['total'] - ml_pred['total'], 2)
                    if ml_pred else None
                ),
                'strategic_vs_hybrid': round(
                    strategic_pred['total'] - hybrid_pred['prediction']['total'], 2
                )
            }
        }

        return {
            "message": "Model comparison completed",
            "comparison": comparison
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model comparison error: {str(e)}")


@router.get("/recommendation")
async def get_final_recommendation() -> Dict[str, Any]:
    """
    최종 권고사항 조회
    """
    try:
        recommendation = strategic_dashboard_service._create_final_recommendation()

        # 하이브리드 예측 결과 추가
        hybrid_result = hybrid_prediction_service.predict_wage_increase(2026, 'base')

        final_recommendation = {
            'strategic_recommendation': recommendation,
            'hybrid_prediction': hybrid_result['prediction'],
            'confidence': hybrid_result['confidence'],
            'implementation_guide': {
                'immediate_actions': [
                    '경제지표 모니터링 체계 구축',
                    '경쟁사 임금 동향 분석',
                    'CDMO 수주 실적 추적'
                ],
                'quarterly_review': [
                    'Q1: 최저임금 인상률 확정 반영',
                    'Q2: 상반기 실적 반영 조정',
                    'Q3: 하반기 전망 반영',
                    'Q4: 최종 결정 및 공지'
                ]
            }
        }

        return {
            "message": "Recommendation generated successfully",
            "recommendation": final_recommendation
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Recommendation error: {str(e)}")
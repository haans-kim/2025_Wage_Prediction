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


@router.get("/historical")
async def get_historical_data() -> Dict[str, Any]:
    """
    과거 임금인상률 데이터 조회
    """
    try:
        historical_data = [
            {"year": 2020, "actual_increase": 0.032, "base_up": 0.020},
            {"year": 2021, "actual_increase": 0.038, "base_up": 0.025},
            {"year": 2022, "actual_increase": 0.045, "base_up": 0.030},
            {"year": 2023, "actual_increase": 0.052, "base_up": 0.035},
            {"year": 2024, "actual_increase": 0.048, "base_up": 0.032},
            {"year": 2025, "actual_increase": 0.054, "base_up": 0.036},
            {"year": 2026, "actual_increase": 0.040, "base_up": 0.024}  # 예측값
        ]
        return {"data": historical_data}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Historical data error: {str(e)}")


@router.get("/scenarios")
async def get_scenarios() -> Dict[str, Any]:
    """
    시나리오별 예측 결과 조회
    """
    try:
        scenarios = [
            {
                "name": "보수적 시나리오",
                "description": "경기 둔화 및 보수적 접근",
                "total_increase": 0.035,
                "base_increase": 0.022,
                "performance_increase": 0.013,
                "variables": {
                    "gdp_growth": 1.5,
                    "inflation_rate": 1.2,
                    "unemployment_rate": 4.0,
                    "industry_growth": 2.0,
                    "company_performance": 5.0,
                    "labor_union_power": 60
                }
            },
            {
                "name": "기본 시나리오",
                "description": "현재 경제 전망 기준",
                "total_increase": 0.040,
                "base_increase": 0.024,
                "performance_increase": 0.016,
                "variables": {
                    "gdp_growth": 2.2,
                    "inflation_rate": 1.8,
                    "unemployment_rate": 3.5,
                    "industry_growth": 3.0,
                    "company_performance": 8.0,
                    "labor_union_power": 65
                }
            },
            {
                "name": "낙관적 시나리오",
                "description": "경기 호조 및 성과 우수",
                "total_increase": 0.048,
                "base_increase": 0.028,
                "performance_increase": 0.020,
                "variables": {
                    "gdp_growth": 3.0,
                    "inflation_rate": 2.2,
                    "unemployment_rate": 3.0,
                    "industry_growth": 4.5,
                    "company_performance": 12.0,
                    "labor_union_power": 70
                }
            }
        ]
        return {"scenarios": scenarios}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Scenarios error: {str(e)}")


@router.get("/sensitivity")
async def get_sensitivity_analysis() -> Dict[str, Any]:
    """
    민감도 분석 결과 조회
    """
    try:
        analysis = [
            {"variable": "GDP 성장률", "impact": 0.8, "direction": "positive"},
            {"variable": "인플레이션율", "impact": 0.6, "direction": "positive"},
            {"variable": "실업률", "impact": -0.4, "direction": "negative"},
            {"variable": "산업성장률", "impact": 0.7, "direction": "positive"},
            {"variable": "회사실적", "impact": 0.9, "direction": "positive"},
            {"variable": "노조협상력", "impact": 0.5, "direction": "positive"}
        ]
        return {"analysis": analysis}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Sensitivity analysis error: {str(e)}")


@router.get("/feature-importance")
async def get_feature_importance() -> Dict[str, Any]:
    """
    실제 데이터 기반 Feature Importance 조회
    """
    try:
        from app.services.data_service import data_service
        from app.services.modeling_service import modeling_service
        from app.services.analysis_service import analysis_service

        # 1. 먼저 학습된 모델의 Feature Importance 시도
        if modeling_service.current_model is not None:
            try:
                importance_data = analysis_service.get_feature_importance(
                    model=modeling_service.current_model,
                    method='permutation',
                    top_n=15
                )
                if importance_data and 'feature_importance' in importance_data:
                    features = []
                    for item in importance_data['feature_importance']:
                        features.append({
                            "name": item['feature'],
                            "korean_name": item.get('feature_korean', item['feature']),
                            "importance": item['importance']
                        })
                    if features:  # 비어있지 않으면 반환
                        return {"features": features}
            except:
                pass  # 실패하면 다음 방법으로

        # 2. 실제 데이터 컬럼 기반으로 상관관계 계산
        if data_service.current_data is not None:
            # 실제 데이터의 feature 컬럼 사용
            columns = list(data_service.current_data.columns)

            # 타겟 컬럼 찾기
            target_column = None
            for col in reversed(columns):
                if 'increase' in col.lower() or 'rate' in col.lower() or '인상' in col:
                    target_column = col
                    break

            # 타겟과의 상관관계 계산
            if target_column and target_column in data_service.current_data.columns:
                correlations = data_service.current_data.corr()[target_column].abs()
                correlations = correlations.drop(target_column, errors='ignore')

                # wage_increase 관련 컬럼들 제외 (타겟의 구성요소이므로)
                exclude_columns = [
                    'wage_increase_bu_sbl',
                    'wage_increase_mi_sbl',
                    'wage_increase_total_sbl',
                    'wage_increase_bu_group',
                    'wage_increase_mi_group',
                    'wage_increase_total_group'
                ]
                for col in exclude_columns:
                    correlations = correlations.drop(col, errors='ignore')

                correlations = correlations.sort_values(ascending=False)

                # 한글 매핑
                korean_names = {
                    "wage_increase_bu_group": "업계 Base-up 인상률",
                    "wage_increase_mi_group": "업계 MI 인상률",
                    "wage_increase_total_group": "업계 총 인상률",
                    "gdp_growth_kr": "한국 GDP 성장률",
                    "cpi_kr": "한국 소비자물가지수",
                    "unemployment_rate_kr": "한국 실업률",
                    "minimum_wage_increase_kr": "최저임금 인상률",
                    "gdp_growth_usa": "미국 GDP 성장률",
                    "cpi_usa": "미국 소비자물가지수",
                    "eci_usa": "미국 고용비용지수",
                    "exchange_rate_change_krw": "환율 변동률",
                    "revenue_growth_sbl": "매출 성장률",
                    "op_profit_growth_sbl": "영업이익 성장률",
                    "labor_to_revenue_sbl": "인건비 비중",
                    "labor_cost_per_employee_sbl": "인당 인건비",
                    "revenue_per_employee_sbl": "인당 매출액",
                    "op_profit_per_employee_sbl": "인당 영업이익",
                    "hcroi_sbl": "인적자본 투자수익률",
                    "hcva_sbl": "인적자본 부가가치",
                    "market_size_growth_rate": "시장규모 성장률",
                    "compensation_competitiveness": "보상 경쟁력",
                    "wage_increase_ce": "CE 임금인상률",
                    "public_sector_wage_increase": "공공부문 임금인상률"
                }

                # 상위 10개 feature 선택
                top_features = correlations.head(10)
                features = []

                # 중요도 정규화 (합이 1이 되도록)
                total_corr = top_features.sum()

                for feature_name, correlation in top_features.items():
                    importance = correlation / total_corr if total_corr > 0 else 0.1
                    features.append({
                        "name": feature_name,
                        "korean_name": korean_names.get(feature_name, feature_name),
                        "importance": float(importance)
                    })

                return {"features": features}

        # 최종 폴백: 기본값
        default_features = [
            {"name": "wage_increase_bu_group", "korean_name": "업계 Base-up 인상률", "importance": 0.25},
            {"name": "gdp_growth_kr", "korean_name": "한국 GDP 성장률", "importance": 0.20},
            {"name": "cpi_kr", "korean_name": "한국 소비자물가지수", "importance": 0.18},
            {"name": "market_size_growth_rate", "korean_name": "시장규모 성장률", "importance": 0.15},
            {"name": "hcroi_sbl", "korean_name": "인적자본 투자수익률", "importance": 0.12},
            {"name": "unemployment_rate_kr", "korean_name": "한국 실업률", "importance": 0.10}
        ]
        return {"features": default_features}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Feature importance error: {str(e)}")


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
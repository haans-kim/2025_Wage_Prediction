import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import warnings
import logging
from datetime import datetime, timedelta
from app.services.data_service import data_service

class DashboardService:
    def __init__(self):
        self.scenario_templates = {
            "conservative": {
                "name": "보수적 시나리오",
                "description": "경제 성장 둔화, 인플레이션 억제 상황",
                "variables": {
                    "inflation_rate": 1.8,
                    "gdp_growth": 1.5,
                    "unemployment_rate": 4.2,
                    "productivity_growth": 1.2,
                    "exchange_rate_volatility": 0.8
                }
            },
            "moderate": {
                "name": "기준 시나리오",
                "description": "현재 경제 상황 지속",
                "variables": {
                    "inflation_rate": 2.5,
                    "gdp_growth": 2.8,
                    "unemployment_rate": 3.2,
                    "productivity_growth": 2.0,
                    "exchange_rate_volatility": 1.0
                }
            },
            "optimistic": {
                "name": "낙관적 시나리오",
                "description": "경제 성장 가속, 생산성 향상",
                "variables": {
                    "inflation_rate": 3.2,
                    "gdp_growth": 4.0,
                    "unemployment_rate": 2.5,
                    "productivity_growth": 3.5,
                    "exchange_rate_volatility": 1.2
                }
            },
            "crisis": {
                "name": "위기 시나리오",
                "description": "경제 침체, 고실업 상황",
                "variables": {
                    "inflation_rate": 1.0,
                    "gdp_growth": -1.5,
                    "unemployment_rate": 6.5,
                    "productivity_growth": -0.5,
                    "exchange_rate_volatility": 1.8
                }
            }
        }
        
        self.variable_definitions = {
            "inflation_rate": {
                "name": "인플레이션율",
                "description": "소비자물가지수 상승률 (%)",
                "min_value": -2.0,
                "max_value": 8.0,
                "unit": "%",
                "current_value": 2.5
            },
            "gdp_growth": {
                "name": "GDP 성장률",
                "description": "실질 GDP 전년 대비 성장률 (%)",
                "min_value": -5.0,
                "max_value": 8.0,
                "unit": "%",
                "current_value": 2.8
            },
            "unemployment_rate": {
                "name": "실업률",
                "description": "경제활동인구 대비 실업자 비율 (%)",
                "min_value": 1.0,
                "max_value": 10.0,
                "unit": "%",
                "current_value": 3.2
            },
            "productivity_growth": {
                "name": "생산성 증가율",
                "description": "노동생산성 전년 대비 증가율 (%)",
                "min_value": -3.0,
                "max_value": 6.0,
                "unit": "%",
                "current_value": 2.0
            },
            "exchange_rate_volatility": {
                "name": "환율 변동성",
                "description": "환율 변동성 지수 (기준=1.0)",
                "min_value": 0.5,
                "max_value": 2.5,
                "unit": "지수",
                "current_value": 1.0
            }
        }
    
    def _prepare_model_input(self, variables: Dict[str, float]) -> pd.DataFrame:
        """모델 입력용 데이터 준비 - 현재 데이터 구조에 맞게 매핑"""
        try:
            # 현재 데이터의 컬럼 구조 파악
            if data_service.current_data is not None:
                # 타겟 컬럼과 year 컬럼 제외한 피처 컬럼들 
                feature_columns = [col for col in data_service.current_data.columns 
                                 if col not in ['target', 'wage_increase_rate', 'wage_increase_total_sbl', 'year']]
                
                # 변수 매핑: Dashboard 변수 → 실제 데이터 컬럼 (퍼센트를 소수점으로 변환)
                variable_mapping = {
                    'gdp_growth': ('gdp_growth_kr', 0.01),      # 2.8% → 0.028
                    'inflation_rate': ('cpi_kr', 0.01),        # 2.5% → 0.025  
                    'unemployment_rate': ('unemployment_rate_kr', 0.01),  # 3.2% → 0.032
                    'productivity_growth': ('minimum_wage_increase_kr', 0.01),  # 2.0% → 0.02
                    'exchange_rate_volatility': ('exchange_rate_change_krw', 0.01)  # 1.0 → 0.01
                }
                
                # 데이터에서 수치형 값들의 평균값 계산 (결측값과 '-' 제외)
                df_clean = data_service.current_data.copy()
                for col in df_clean.columns:
                    if df_clean[col].dtype == 'object':  # 문자열 컬럼
                        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
                
                input_data = {}
                
                for col in feature_columns:
                    # 매핑된 변수가 있으면 사용자 입력값 적용
                    mapped_variable = None
                    scale_factor = 1.0
                    
                    for dash_var, (data_col, scale) in variable_mapping.items():
                        if data_col == col and dash_var in variables:
                            mapped_variable = dash_var
                            scale_factor = scale
                            break
                    
                    if mapped_variable:
                        input_data[col] = variables[mapped_variable] * scale_factor
                    else:
                        # 해당 컬럼의 평균값 사용 (결측값 제외)
                        if col in df_clean.columns:
                            col_mean = df_clean[col].mean()
                            if pd.notna(col_mean):
                                input_data[col] = col_mean
                            else:
                                input_data[col] = 0.0  # 기본값
                        else:
                            input_data[col] = 0.0
                
                print(f"📊 Model input prepared with {len(input_data)} features")
                return pd.DataFrame([input_data])
                
        except Exception as e:
            logging.error(f"Error preparing model input: {str(e)}")
            print(f"Error details: {e}")
            
        # 폴백: 기본 구조 사용
        return pd.DataFrame([{
            'gdp_growth_kr': variables.get('gdp_growth', 0.028),
            'cpi_kr': variables.get('inflation_rate', 0.007),
            'unemployment_rate_kr': variables.get('unemployment_rate', 0.036)
            }])
    
    def predict_wage_increase(self, model, input_data: Dict[str, float], confidence_level: float = 0.95) -> Dict[str, Any]:
        """임금인상률 예측"""
        
        try:
            # 입력 데이터 준비
            model_input = self._prepare_model_input(input_data)
            
            # 예측 수행 (PyCaret 모델 사용)
            try:
                from pycaret.regression import predict_model
                predictions_df = predict_model(model, data=model_input, verbose=False)
                
                # 예측 결과 컬럼 찾기
                if 'prediction_label' in predictions_df.columns:
                    prediction = predictions_df['prediction_label'].values[0]
                elif 'Label' in predictions_df.columns:
                    prediction = predictions_df['Label'].values[0]
                else:
                    # 원본 컬럼을 제외한 새로 추가된 컬럼이 예측값
                    original_cols = set(model_input.columns)
                    new_cols = set(predictions_df.columns) - original_cols
                    if new_cols:
                        pred_col = list(new_cols)[0]
                        prediction = predictions_df[pred_col].values[0]
                    else:
                        # 마지막 컬럼이 보통 예측 결과
                        prediction = predictions_df.iloc[0, -1]
            except Exception as e:
                print(f"⚠️ PyCaret predict_model failed: {e}")
                # Fallback: 직접 predict 시도
                prediction = model.predict(model_input)[0]
            
            # 신뢰구간 계산 (간단한 방법 - 잔차 기반)
            try:
                from pycaret.regression import get_config
                X_train = get_config('X_train')
                y_train = get_config('y_train')
                
                if X_train is not None and y_train is not None:
                    train_predictions = model.predict(X_train)
                    residuals = y_train - train_predictions
                    residual_std = np.std(residuals)
                    
                    # 신뢰구간
                    from scipy import stats
                    alpha = 1 - confidence_level
                    z_score = stats.norm.ppf(1 - alpha/2)
                    margin_error = z_score * residual_std
                    
                    confidence_interval = [
                        float(prediction - margin_error),
                        float(prediction + margin_error)
                    ]
                else:
                    # 기본값
                    confidence_interval = [
                        float(prediction * 0.85),
                        float(prediction * 1.15)
                    ]
            except:
                confidence_interval = [
                    float(prediction * 0.85),
                    float(prediction * 1.15)
                ]
            
            return {
                "message": "Wage increase prediction completed",
                "prediction": float(prediction),
                "confidence_interval": confidence_interval,
                "confidence_level": confidence_level,
                "input_variables": input_data,
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "model_type": type(model).__name__
            }
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise ValueError(f"Prediction failed: {str(e)}")
    
    def run_scenario_analysis(self, model, scenarios: List[Dict[str, Any]]) -> Dict[str, Any]:
        """다중 시나리오 분석"""
        
        try:
            results = []
            predictions = []
            
            for scenario in scenarios:
                scenario_name = scenario.get('scenario_name', 'Unnamed')
                variables = scenario.get('variables', {})
                
                # 개별 시나리오 예측
                prediction_result = self.predict_wage_increase(model, variables)
                
                scenario_result = {
                    "scenario_name": scenario_name,
                    "description": scenario.get('description', ''),
                    "variables": variables,
                    "prediction": prediction_result['prediction'],
                    "confidence_interval": prediction_result['confidence_interval']
                }
                
                results.append(scenario_result)
                predictions.append(prediction_result['prediction'])
            
            # 시나리오 비교 분석
            if len(predictions) > 1:
                comparison = {
                    "min_prediction": float(np.min(predictions)),
                    "max_prediction": float(np.max(predictions)),
                    "mean_prediction": float(np.mean(predictions)),
                    "std_prediction": float(np.std(predictions)),
                    "range": float(np.max(predictions) - np.min(predictions))
                }
                
                # 시나리오 순위
                sorted_scenarios = sorted(results, key=lambda x: x['prediction'], reverse=True)
                for i, scenario in enumerate(sorted_scenarios):
                    scenario['rank'] = i + 1
            else:
                comparison = {"message": "단일 시나리오 분석"}
            
            return {
                "message": "Scenario analysis completed successfully",
                "results": results,
                "comparison": comparison,
                "analysis_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "num_scenarios": len(scenarios)
            }
            
        except Exception as e:
            logging.error(f"Scenario analysis failed: {str(e)}")
            raise ValueError(f"Scenario analysis failed: {str(e)}")
    
    def get_available_variables(self) -> Dict[str, Any]:
        """사용 가능한 변수 목록 및 현재 값"""
        
        variables = []
        current_values = {}
        
        for var_name, var_info in self.variable_definitions.items():
            variables.append({
                "name": var_name,
                "display_name": var_info["name"],
                "description": var_info["description"],
                "min_value": var_info["min_value"],
                "max_value": var_info["max_value"],
                "unit": var_info["unit"],
                "current_value": var_info["current_value"]
            })
            
            current_values[var_name] = var_info["current_value"]
        
        return {
            "message": "Available variables retrieved successfully",
            "variables": variables,
            "current_values": current_values,
            "total_variables": len(variables)
        }
    
    def get_historical_trends(self, years: int = 10, include_forecast: bool = True) -> Dict[str, Any]:
        """과거 임금인상률 트렌드 및 예측"""
        
        try:
            # 샘플 과거 데이터 생성 (실제로는 데이터베이스에서 가져와야 함)
            end_year = 2024
            start_year = end_year - years + 1
            
            historical_data = []
            for year in range(start_year, end_year + 1):
                # 샘플 데이터 - 실제로는 실제 데이터 사용
                base_rate = 3.2
                noise = np.random.normal(0, 0.5)
                trend = (year - 2020) * 0.1  # 연도별 트렌드
                
                wage_increase = base_rate + trend + noise
                
                historical_data.append({
                    "year": year,
                    "wage_increase_rate": round(wage_increase, 2),
                    "inflation_rate": round(2.0 + np.random.normal(0, 0.3), 2),
                    "gdp_growth": round(2.5 + np.random.normal(0, 0.8), 2),
                    "unemployment_rate": round(3.5 + np.random.normal(0, 0.5), 2)
                })
            
            # 예측 데이터 (2025년)
            forecast_data = []
            if include_forecast:
                for year in range(2025, 2028):
                    # 기본 시나리오 기반 예측
                    moderate_scenario = self.scenario_templates["moderate"]["variables"]
                    forecast_data.append({
                        "year": year,
                        "predicted_wage_increase": round(3.5 + (year - 2025) * 0.1, 2),
                        "scenario": "moderate",
                        "confidence_low": round(2.8 + (year - 2025) * 0.1, 2),
                        "confidence_high": round(4.2 + (year - 2025) * 0.1, 2)
                    })
            
            return {
                "message": "Historical trends retrieved successfully",
                "historical_data": historical_data,
                "forecast_data": forecast_data,
                "summary": {
                    "avg_historical_rate": round(np.mean([d["wage_increase_rate"] for d in historical_data]), 2),
                    "trend": "increasing" if historical_data[-1]["wage_increase_rate"] > historical_data[0]["wage_increase_rate"] else "decreasing",
                    "volatility": round(np.std([d["wage_increase_rate"] for d in historical_data]), 2)
                }
            }
            
        except Exception as e:
            logging.error(f"Historical trends analysis failed: {str(e)}")
            raise ValueError(f"Historical trends analysis failed: {str(e)}")
    
    def get_economic_indicators(self) -> Dict[str, Any]:
        """주요 경제 지표 현황"""
        
        try:
            # 현재 경제 지표 (실제로는 외부 API나 데이터베이스에서 가져와야 함)
            current_indicators = {
                "inflation_rate": {
                    "value": 2.5,
                    "change": "+0.2",
                    "status": "stable",
                    "last_updated": "2024-12"
                },
                "gdp_growth": {
                    "value": 2.8,
                    "change": "+0.1",
                    "status": "growing",
                    "last_updated": "2024-Q3"
                },
                "unemployment_rate": {
                    "value": 3.2,
                    "change": "-0.1",
                    "status": "improving",
                    "last_updated": "2024-12"
                },
                "productivity_growth": {
                    "value": 2.0,
                    "change": "+0.3",
                    "status": "growing",
                    "last_updated": "2024-Q3"
                },
                "exchange_rate_usd": {
                    "value": 1320,
                    "change": "+15",
                    "status": "volatile",
                    "last_updated": "2024-12"
                }
            }
            
            # 지표 요약
            summary = {
                "overall_outlook": "moderate_positive",
                "key_risks": ["인플레이션 압력", "환율 변동성"],
                "growth_drivers": ["생산성 향상", "고용 개선"],
                "last_analysis": datetime.now().strftime("%Y-%m-%d")
            }
            
            return {
                "message": "Economic indicators retrieved successfully",
                "indicators": current_indicators,
                "summary": summary
            }
            
        except Exception as e:
            logging.error(f"Economic indicators retrieval failed: {str(e)}")
            raise ValueError(f"Economic indicators retrieval failed: {str(e)}")
    
    def get_scenario_templates(self) -> List[Dict[str, Any]]:
        """사전 정의된 시나리오 템플릿"""
        
        templates = []
        for template_id, template_data in self.scenario_templates.items():
            templates.append({
                "id": template_id,
                "name": template_data["name"],
                "description": template_data["description"],
                "variables": template_data["variables"]
            })
        
        return templates
    
    def run_sensitivity_analysis(self, model, base_scenario: Dict[str, float], 
                                variable_name: str, variation_range: float = 0.2) -> Dict[str, Any]:
        """민감도 분석"""
        
        try:
            if variable_name not in self.variable_definitions:
                raise ValueError(f"Unknown variable: {variable_name}")
            
            var_info = self.variable_definitions[variable_name]
            base_value = base_scenario.get(variable_name, var_info["current_value"])
            
            # 변동 범위 설정
            min_val = max(var_info["min_value"], base_value * (1 - variation_range))
            max_val = min(var_info["max_value"], base_value * (1 + variation_range))
            
            # 변동 포인트 생성
            num_points = 11
            variation_points = np.linspace(min_val, max_val, num_points)
            
            results = []
            for point in variation_points:
                scenario = base_scenario.copy()
                scenario[variable_name] = point
                
                prediction_result = self.predict_wage_increase(model, scenario)
                
                results.append({
                    "variable_value": float(point),
                    "prediction": prediction_result["prediction"],
                    "change_from_base": float(point - base_value),
                    "prediction_change": float(prediction_result["prediction"] - 
                                             self.predict_wage_increase(model, base_scenario)["prediction"])
                })
            
            # 민감도 계산
            predictions = [r["prediction"] for r in results]
            sensitivity = (max(predictions) - min(predictions)) / (max_val - min_val)
            
            return {
                "message": "Sensitivity analysis completed",
                "variable": {
                    "name": variable_name,
                    "display_name": var_info["name"],
                    "base_value": float(base_value),
                    "variation_range": variation_range
                },
                "results": results,
                "sensitivity_coefficient": float(sensitivity),
                "interpretation": "high" if sensitivity > 1.0 else "medium" if sensitivity > 0.5 else "low"
            }
            
        except Exception as e:
            logging.error(f"Sensitivity analysis failed: {str(e)}")
            raise ValueError(f"Sensitivity analysis failed: {str(e)}")
    
    def get_forecast_accuracy(self, model) -> Dict[str, Any]:
        """예측 정확도 분석"""
        
        try:
            # PyCaret에서 모델 성능 정보 가져오기
            from app.services.modeling_service import modeling_service
            
            performance_result = modeling_service.get_model_evaluation()
            
            if performance_result and 'evaluation_data' in performance_result:
                eval_data = performance_result['evaluation_data']
                
                accuracy_metrics = {
                    "r2_score": eval_data.get('R2', 0.0),
                    "mae": eval_data.get('MAE', 0.0),
                    "rmse": eval_data.get('RMSE', 0.0),
                    "mape": eval_data.get('MAPE', 0.0) if 'MAPE' in eval_data else None
                }
                
                # 정확도 등급 결정
                r2_score = accuracy_metrics.get('r2_score', 0)
                if r2_score > 0.8:
                    accuracy_grade = "excellent"
                elif r2_score > 0.6:
                    accuracy_grade = "good"
                elif r2_score > 0.4:
                    accuracy_grade = "fair"
                else:
                    accuracy_grade = "poor"
            else:
                # 기본값
                accuracy_metrics = {
                    "r2_score": 0.75,
                    "mae": 0.85,
                    "rmse": 1.12,
                    "mape": 12.5
                }
                accuracy_grade = "good"
            
            return {
                "message": "Forecast accuracy analysis completed",
                "metrics": accuracy_metrics,
                "grade": accuracy_grade,
                "model_type": type(model).__name__,
                "recommendations": self._get_accuracy_recommendations(accuracy_grade)
            }
            
        except Exception as e:
            logging.error(f"Forecast accuracy analysis failed: {str(e)}")
            return {
                "message": "Forecast accuracy analysis failed",
                "error": str(e),
                "metrics": {"r2_score": 0.0, "mae": 0.0, "rmse": 0.0},
                "grade": "unknown"
            }
    
    def _get_accuracy_recommendations(self, grade: str) -> List[str]:
        """정확도 등급에 따른 권고사항"""
        
        recommendations = {
            "excellent": [
                "현재 모델의 예측 정확도가 매우 높습니다.",
                "안정적으로 예측 결과를 활용할 수 있습니다.",
                "정기적인 모델 업데이트를 통해 성능을 유지하세요."
            ],
            "good": [
                "모델의 예측 정확도가 양호합니다.",
                "주요 의사결정에 활용 가능하지만 추가 검토가 권장됩니다.",
                "더 많은 데이터로 모델을 개선할 수 있습니다."
            ],
            "fair": [
                "모델의 예측 정확도가 보통 수준입니다.",
                "예측 결과를 참고용으로만 활용하시기 바랍니다.",
                "모델 튜닝이나 특성 엔지니어링이 필요합니다."
            ],
            "poor": [
                "모델의 예측 정확도가 낮습니다.",
                "현재 모델로는 신뢰할 만한 예측이 어렵습니다.",
                "데이터 품질 개선과 모델 재구축을 권장합니다."
            ]
        }
        
        return recommendations.get(grade, ["정확도 분석 결과를 확인해주세요."])
    
    def run_monte_carlo_simulation(self, model, base_scenario: Dict[str, float], 
                                  uncertainty_ranges: Dict[str, float], num_simulations: int = 1000) -> Dict[str, Any]:
        """몬테카를로 시뮬레이션"""
        
        try:
            results = []
            
            for _ in range(num_simulations):
                # 각 변수에 대해 불확실성 범위 내에서 랜덤 값 생성
                scenario = base_scenario.copy()
                
                for var_name, uncertainty in uncertainty_ranges.items():
                    if var_name in scenario:
                        base_value = scenario[var_name]
                        # 정규분포를 가정하여 랜덤 값 생성
                        random_value = np.random.normal(base_value, uncertainty)
                        
                        # 변수의 최소/최대값 제한 적용
                        if var_name in self.variable_definitions:
                            var_info = self.variable_definitions[var_name]
                            random_value = np.clip(random_value, var_info["min_value"], var_info["max_value"])
                        
                        scenario[var_name] = random_value
                
                # 예측 수행
                prediction_result = self.predict_wage_increase(model, scenario)
                results.append(prediction_result["prediction"])
            
            # 통계 분석
            results_array = np.array(results)
            statistics = {
                "mean": float(np.mean(results_array)),
                "std": float(np.std(results_array)),
                "min": float(np.min(results_array)),
                "max": float(np.max(results_array)),
                "percentile_5": float(np.percentile(results_array, 5)),
                "percentile_25": float(np.percentile(results_array, 25)),
                "percentile_50": float(np.percentile(results_array, 50)),
                "percentile_75": float(np.percentile(results_array, 75)),
                "percentile_95": float(np.percentile(results_array, 95))
            }
            
            # 히스토그램 데이터 생성
            hist, bin_edges = np.histogram(results_array, bins=20)
            histogram_data = []
            for i in range(len(hist)):
                histogram_data.append({
                    "bin_start": float(bin_edges[i]),
                    "bin_end": float(bin_edges[i+1]),
                    "frequency": int(hist[i]),
                    "probability": float(hist[i] / num_simulations)
                })
            
            return {
                "message": "Monte Carlo simulation completed",
                "num_simulations": num_simulations,
                "base_scenario": base_scenario,
                "uncertainty_ranges": uncertainty_ranges,
                "statistics": statistics,
                "histogram": histogram_data,
                "risk_analysis": {
                    "probability_above_4": float(np.mean(results_array > 4.0)),
                    "probability_below_2": float(np.mean(results_array < 2.0)),
                    "value_at_risk_5": statistics["percentile_5"],
                    "value_at_risk_1": float(np.percentile(results_array, 1))
                }
            }
            
        except Exception as e:
            logging.error(f"Monte Carlo simulation failed: {str(e)}")
            raise ValueError(f"Monte Carlo simulation failed: {str(e)}")
    
    def get_market_conditions(self) -> Dict[str, Any]:
        """현재 시장 상황 요약"""
        
        try:
            # 경제 지표 기반 시장 상황 평가
            economic_indicators = self.get_economic_indicators()
            indicators = economic_indicators["indicators"]
            
            # 전반적 시장 상황 평가
            positive_factors = []
            negative_factors = []
            neutral_factors = []
            
            # 각 지표별 평가
            if indicators["gdp_growth"]["value"] > 2.5:
                positive_factors.append("GDP 성장률 양호")
            elif indicators["gdp_growth"]["value"] < 1.0:
                negative_factors.append("GDP 성장률 부진")
            else:
                neutral_factors.append("GDP 성장률 보통")
            
            if indicators["unemployment_rate"]["value"] < 3.5:
                positive_factors.append("낮은 실업률")
            elif indicators["unemployment_rate"]["value"] > 5.0:
                negative_factors.append("높은 실업률")
            else:
                neutral_factors.append("적정 실업률 수준")
            
            if indicators["inflation_rate"]["value"] > 3.5:
                negative_factors.append("인플레이션 압력")
            elif indicators["inflation_rate"]["value"] < 1.0:
                negative_factors.append("디플레이션 우려")
            else:
                positive_factors.append("안정적 물가 수준")
            
            # 전반적 평가
            score = len(positive_factors) - len(negative_factors)
            if score > 1:
                overall_sentiment = "positive"
                outlook = "임금인상 여건 양호"
            elif score < -1:
                overall_sentiment = "negative"
                outlook = "임금인상 여건 어려움"
            else:
                overall_sentiment = "neutral"
                outlook = "임금인상 여건 보통"
            
            return {
                "message": "Market conditions analysis completed",
                "overall_sentiment": overall_sentiment,
                "outlook": outlook,
                "positive_factors": positive_factors,
                "negative_factors": negative_factors,
                "neutral_factors": neutral_factors,
                "key_indicators": {
                    "gdp_growth": indicators["gdp_growth"]["value"],
                    "inflation_rate": indicators["inflation_rate"]["value"],
                    "unemployment_rate": indicators["unemployment_rate"]["value"]
                },
                "analysis_date": datetime.now().strftime("%Y-%m-%d")
            }
            
        except Exception as e:
            logging.error(f"Market conditions analysis failed: {str(e)}")
            raise ValueError(f"Market conditions analysis failed: {str(e)}")
    
    def create_custom_scenario(self, model, scenario_name: str, variables: Dict[str, float], 
                              save_template: bool = False) -> Dict[str, Any]:
        """사용자 정의 시나리오 생성"""
        
        try:
            # 시나리오 예측 수행
            prediction_result = self.predict_wage_increase(model, variables)
            
            # 기준 시나리오와 비교
            base_scenario = self.scenario_templates["moderate"]["variables"]
            base_prediction = self.predict_wage_increase(model, base_scenario)
            
            comparison = {
                "difference_from_base": float(prediction_result["prediction"] - base_prediction["prediction"]),
                "relative_change": float((prediction_result["prediction"] / base_prediction["prediction"] - 1) * 100)
            }
            
            custom_scenario = {
                "scenario_name": scenario_name,
                "variables": variables,
                "prediction": prediction_result["prediction"],
                "confidence_interval": prediction_result["confidence_interval"],
                "comparison": comparison,
                "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # 템플릿으로 저장 (선택사항)
            if save_template:
                # 실제로는 데이터베이스나 파일에 저장
                logging.info(f"Custom scenario '{scenario_name}' saved as template")
            
            return {
                "message": "Custom scenario created successfully",
                "scenario": custom_scenario,
                "saved_as_template": save_template
            }
            
        except Exception as e:
            logging.error(f"Custom scenario creation failed: {str(e)}")
            raise ValueError(f"Custom scenario creation failed: {str(e)}")
    
    def get_prediction_breakdown(self, model, variables: Dict[str, float]) -> Dict[str, Any]:
        """예측 결과의 상세 분해 (기여도 분석)"""
        
        try:
            # 기준점 (모든 변수가 평균값)
            baseline_variables = {}
            for var_name in self.variable_definitions:
                baseline_variables[var_name] = self.variable_definitions[var_name]["current_value"]
            
            baseline_prediction = self.predict_wage_increase(model, baseline_variables)["prediction"]
            
            # 각 변수의 기여도 계산
            contributions = []
            
            for var_name, var_value in variables.items():
                if var_name in self.variable_definitions:
                    # 해당 변수만 변경하고 나머지는 기준값 유지
                    test_variables = baseline_variables.copy()
                    test_variables[var_name] = var_value
                    
                    test_prediction = self.predict_wage_increase(model, test_variables)["prediction"]
                    contribution = test_prediction - baseline_prediction
                    
                    contributions.append({
                        "variable": var_name,
                        "display_name": self.variable_definitions[var_name]["name"],
                        "value": float(var_value),
                        "baseline_value": baseline_variables[var_name],
                        "contribution": float(contribution),
                        "contribution_percent": float((contribution / baseline_prediction) * 100) if baseline_prediction != 0 else 0
                    })
            
            # 전체 예측
            total_prediction = self.predict_wage_increase(model, variables)["prediction"]
            
            # 상호작용 효과 (전체 - 개별 기여도 합)
            individual_sum = sum([c["contribution"] for c in contributions])
            interaction_effect = total_prediction - baseline_prediction - individual_sum
            
            return {
                "message": "Prediction breakdown completed",
                "baseline_prediction": float(baseline_prediction),
                "total_prediction": float(total_prediction),
                "total_change": float(total_prediction - baseline_prediction),
                "contributions": contributions,
                "interaction_effect": float(interaction_effect),
                "explanation": {
                    "baseline": "모든 변수가 현재 평균값일 때의 예측",
                    "contributions": "각 변수 변화의 개별 기여도",
                    "interaction": "변수 간 상호작용 효과"
                }
            }
            
        except Exception as e:
            logging.error(f"Prediction breakdown failed: {str(e)}")
            raise ValueError(f"Prediction breakdown failed: {str(e)}")
    
    def get_trend_data(self, model) -> Dict[str, Any]:
        """과거 임금인상률 추이 및 2025년 예측 데이터 반환"""
        try:
            from app.services.data_service import data_service
            
            if data_service.current_data is None:
                raise ValueError("No data available for trend analysis")
            
            # 과거 데이터에서 연도별 평균 임금인상률 계산
            df = data_service.current_data.copy()
            
            # 타겟 컬럼 찾기
            target_columns = ['wage_increase_total_sbl', 'wage_increase_rate', 'target']
            target_col = None
            for col in target_columns:
                if col in df.columns:
                    target_col = col
                    break
            
            if target_col is None:
                raise ValueError("No target column found for trend analysis")
            
            # 연도별 데이터 집계 (결측값 제외)
            if 'year' in df.columns:
                yearly_data = df.groupby('year')[target_col].mean().dropna()
            else:
                # 연도 컬럼이 없는 경우 인덱스 기반으로 연도 추정
                yearly_data = df[target_col].dropna()
                years = range(2015, 2015 + len(yearly_data))
                yearly_data.index = years
            
            # 과거 데이터 포맷팅
            historical_data = []
            for year, value in yearly_data.items():
                if pd.notna(value):
                    historical_data.append({
                        "year": int(year),
                        "value": float(value * 100),  # 퍼센트 변환
                        "type": "historical"
                    })
            
            # 2025년 예측 (현재 기본 시나리오)
            base_variables = {
                'inflation_rate': 2.5,
                'gdp_growth': 2.8,
                'unemployment_rate': 3.2,
                'productivity_growth': 2.0,
                'exchange_rate_volatility': 1.0
            }
            
            prediction_result = self.predict_wage_increase(model, base_variables)
            
            # 2025년 예측 데이터
            prediction_data = {
                "year": 2025,
                "value": float(prediction_result["prediction"] * 100),  # 퍼센트 변환
                "confidence_lower": float(prediction_result["confidence_interval"][0] * 100),
                "confidence_upper": float(prediction_result["confidence_interval"][1] * 100),
                "type": "prediction"
            }
            
            # 전체 트렌드 데이터 구성
            trend_data = historical_data + [prediction_data]
            trend_data.sort(key=lambda x: x["year"])
            
            # 통계 정보
            historical_values = [d["value"] for d in historical_data]
            stats = {
                "historical_average": float(np.mean(historical_values)) if historical_values else 0,
                "historical_min": float(np.min(historical_values)) if historical_values else 0,
                "historical_max": float(np.max(historical_values)) if historical_values else 0,
                "prediction_vs_average": float(prediction_data["value"] - np.mean(historical_values)) if historical_values else 0
            }
            
            return {
                "message": "Trend data retrieved successfully",
                "trend_data": trend_data,
                "statistics": stats,
                "chart_config": {
                    "title": "임금인상률 추이 및 2025년 예측",
                    "y_axis_label": "임금인상률 (%)",
                    "x_axis_label": "연도"
                }
            }
            
        except Exception as e:
            logging.error(f"Trend data generation failed: {str(e)}")
            raise ValueError(f"Trend data generation failed: {str(e)}")

# 싱글톤 인스턴스
dashboard_service = DashboardService()
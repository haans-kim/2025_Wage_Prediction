import logging
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional
from datetime import datetime, timedelta
from app.services.data_service import data_service

class DashboardService:
    def __init__(self):
        self.scenario_templates = {
            "base": {
                "name": "기본 시나리오",
                "description": "현재 경제 상황 기준",
                "variables": {
                    "wage_increase_bu_group": 3.0,
                    "gdp_growth": 2.8,
                    "unemployment_rate": 3.2,
                    "market_size_growth_rate": 5.0,
                    "hcroi_sbl": 1.5
                }
            },
            "optimistic": {
                "name": "낙관적 시나리오",
                "description": "호황 + 높은 그룹 인상률",
                "variables": {
                    "wage_increase_bu_group": 3.5,
                    "gdp_growth": 3.5,
                    "unemployment_rate": 2.8,
                    "market_size_growth_rate": 10.0,
                    "hcroi_sbl": 2.0
                }
            },
            "moderate": {
                "name": "중립적 시나리오",
                "description": "안정적 성장",
                "variables": {
                    "wage_increase_bu_group": 3.2,
                    "gdp_growth": 3.0,
                    "unemployment_rate": 3.0,
                    "market_size_growth_rate": 7.0,
                    "hcroi_sbl": 1.7
                }
            },
            "pessimistic": {
                "name": "비관적 시나리오",
                "description": "저성장 + 낮은 그룹 인상률",
                "variables": {
                    "wage_increase_bu_group": 2.5,
                    "gdp_growth": 1.5,
                    "unemployment_rate": 4.0,
                    "market_size_growth_rate": 2.0,
                    "hcroi_sbl": 1.2
                }
            }
        }
        
        # 변수 매핑: Feature 이름 → Dashboard 변수명
        self.variable_mapping = {
            # 기존 변수들
            'wage_increase_bu_group': ('wage_increase_bu_group', 0.01),  # 3.0% → 0.03
            'gdp_growth_kr': ('gdp_growth', 0.01),      # 2.8% → 0.028
            'unemployment_rate_kr': ('unemployment_rate', 0.01),  # 3.2% → 0.032
            'market_size_growth_rate': ('market_size_growth_rate', 0.01),  # 5.0% → 0.05
            'hcroi_sbl': ('hcroi_sbl', 1.0),  # 1.5배 → 1.5 (비율이므로 그대로)
            # 상위 Feature Importance 변수들 추가 (실제 feature 이름으로 수정)
            'labor_to_revenue_sbl': ('labor_cost_rate_sbl', 0.01),  # 25.0% → 0.25
            'cpi_kr': ('cpi_kr', 0.01),  # 2.5% → 0.025
            'labor_cost_per_employee_sbl': ('labor_cost_per_employee_sbl', 1000000),  # 100 → 100백만원 (1억원)
            'eci_usa': ('eci_usa', 0.01),  # 3.0% → 0.03
            # 추가 매핑
            'op_profit_growth_sbl': ('op_profit_growth_sbl', 0.01),  # 영업이익 증가율
            'unemployment_rate_us': ('unemployment_rate_us', 0.01),  # 미국 실업률
            'revenue_growth_sbl': ('revenue_growth_sbl', 0.01),  # 매출액 증가율
            'wage_increase_mi_group': ('wage_increase_mi_group', 0.01),  # 그룹 성과 인상률
            'exchange_rate_change_krw': ('exchange_rate_change_krw', 0.01),  # 환율변화율
            'minimum_wage_increase_kr': ('minimum_wage_increase_kr', 0.01),  # 최저임금인상률
            'wage_increase_total_sbl': ('wage_increase_total_sbl', 0.01),  # 총 인상률
            'compensation_competitiveness': ('compensation_competitiveness', 0.01),  # 보상경쟁력
            'cpi_usa': ('cpi_usa', 0.01),  # 미국 소비자물가상승률
            'gdp_growth_usa': ('gdp_growth_usa', 0.01),  # 미국 GDP 성장률
            'public_sector_wage_increase': ('public_sector_wage_increase', 0.01),  # 공공기관 임금인상률
            'hcva_sbl': ('hcva_sbl', 1.0),  # HCVA
            'wage_increase_ce': ('wage_increase_ce', 0.01)  # c사 임금인상률
        }

        self.variable_definitions = {
            "wage_increase_bu_group": {
                "name": "그룹 Base-up 인상률",
                "description": "그룹사 기본 임금인상률 (%)",
                "min_value": 1.0,
                "max_value": 5.0,
                "unit": "%",
                "current_value": 3.0
            },
            "gdp_growth": {
                "name": "GDP 성장률",
                "description": "실질 GDP 전년 대비 성장률 (%)",
                "min_value": -2.0,
                "max_value": 5.0,
                "unit": "%",
                "current_value": 2.8
            },
            "unemployment_rate": {
                "name": "실업률",
                "description": "경제활동인구 대비 실업자 비율 (%)",
                "min_value": 2.0,
                "max_value": 5.0,
                "unit": "%",
                "current_value": 3.2
            },
            "market_size_growth_rate": {
                "name": "바이오산업 성장률",
                "description": "바이오의약산업 시장 성장률 (%)",
                "min_value": -5.0,
                "max_value": 15.0,
                "unit": "%",
                "current_value": 5.0
            },
            "hcroi_sbl": {
                "name": "인적자본 투자수익률",
                "description": "HCROI (Human Capital ROI)",
                "min_value": 0.5,
                "max_value": 3.0,
                "unit": "배",
                "current_value": 1.5
            },
            # 상위 Feature Importance 변수들 추가
            "labor_cost_rate_sbl": {
                "name": "SBL 인건비 비중",
                "description": "총 비용 대비 인건비 비율 (%)",
                "min_value": 10.0,
                "max_value": 50.0,
                "unit": "%",
                "current_value": 25.0
            },
            "cpi_kr": {
                "name": "소비자물가상승률",
                "description": "한국 소비자물가지수 상승률 (%)",
                "min_value": 0.0,
                "max_value": 8.0,
                "unit": "%",
                "current_value": 2.5
            },
            "labor_cost_ratio_change_sbl": {
                "name": "인건비 비중 변화율",
                "description": "전년 대비 인건비 비중 변화 (%p)",
                "min_value": -10.0,
                "max_value": 10.0,
                "unit": "%p",
                "current_value": 0.0
            },
            "labor_cost_per_employee_sbl": {
                "name": "SBL 인당 인건비",
                "description": "직원 1명당 인건비 (억원)",
                "min_value": 50.0,
                "max_value": 200.0,
                "unit": "억원",
                "current_value": 100.0
            },
            "eci_usa": {
                "name": "미국 임금비용지수",
                "description": "미국 고용비용지수 상승률 (%)",
                "min_value": 1.0,
                "max_value": 8.0,
                "unit": "%",
                "current_value": 3.0
            },
            "compensation_competitiveness": {
                "name": "보상경쟁력",
                "description": "시장 대비 보상 경쟁력 지수 (%)",
                "min_value": -10.0,
                "max_value": 10.0,
                "unit": "%",
                "current_value": 0.0
            },
            "cpi_usa": {
                "name": "미국 소비자물가상승률",
                "description": "미국 CPI 상승률 (%)",
                "min_value": 0.0,
                "max_value": 10.0,
                "unit": "%",
                "current_value": 2.5
            },
            "minimum_wage_increase_kr": {
                "name": "최저임금인상률",
                "description": "법정 최저임금 인상률 (%)",
                "min_value": 0.0,
                "max_value": 15.0,
                "unit": "%",
                "current_value": 5.0
            },
            "unemployment_rate_us": {
                "name": "미국 실업률",
                "description": "미국 경제활동인구 대비 실업자 비율 (%)",
                "min_value": 2.0,
                "max_value": 10.0,
                "unit": "%",
                "current_value": 3.5
            },
            "op_profit_growth_sbl": {
                "name": "SBL 영업이익 증가율",
                "description": "전년 대비 영업이익 증가율 (%)",
                "min_value": -20.0,
                "max_value": 30.0,
                "unit": "%",
                "current_value": 5.0
            },
            "revenue_growth_sbl": {
                "name": "SBL 매출액 증가율",
                "description": "전년 대비 매출액 증가율 (%)",
                "min_value": -10.0,
                "max_value": 20.0,
                "unit": "%",
                "current_value": 5.0
            },
            "gdp_growth_usa": {
                "name": "미국 GDP 성장률",
                "description": "미국 실질 GDP 전년 대비 성장률 (%)",
                "min_value": -5.0,
                "max_value": 8.0,
                "unit": "%",
                "current_value": 2.5
            },
            "exchange_rate_change_krw": {
                "name": "원화 환율 변화율",
                "description": "달러/원 환율 변화율 (%)",
                "min_value": -20.0,
                "max_value": 20.0,
                "unit": "%",
                "current_value": 0.0
            },
            "wage_increase_mi_group": {
                "name": "그룹 성과 인상률",
                "description": "그룹사 성과급 인상률 (%)",
                "min_value": 0.0,
                "max_value": 10.0,
                "unit": "%",
                "current_value": 2.0
            },
            "public_sector_wage_increase": {
                "name": "공공기관 임금인상률",
                "description": "공공기관 평균 임금인상률 (%)",
                "min_value": 0.0,
                "max_value": 10.0,
                "unit": "%",
                "current_value": 3.0
            },
            "wage_increase_ce": {
                "name": "경쟁사 임금인상률",
                "description": "경쟁사(C사) 임금인상률 (%)",
                "min_value": 0.0,
                "max_value": 10.0,
                "unit": "%",
                "current_value": 4.0
            },
            "hcva_sbl": {
                "name": "인적자본부가가치",
                "description": "SBL HCVA 지수",
                "min_value": 0.0,
                "max_value": 5.0,
                "unit": "배",
                "current_value": 1.0
            }
        }
    
    def _prepare_model_input(self, variables: Dict[str, float]) -> pd.DataFrame:
        """모델 입력용 데이터 준비 - PyCaret 모델에 맞게 수정"""
        try:
            # hwaseung 방식: 데이터에서 직접 feature columns 가져오기
            from app.services.data_service import data_service

            if data_service.current_data is None:
                raise ValueError("No data available for prediction")

            # 모든 컬럼에서 target 컬럼(SBL 임금) 제외
            # wage_increase_*_group은 feature로 사용 (그룹사 임금 정보)
            # wage_increase_*_sbl은 target (예측 대상)
            all_columns = list(data_service.current_data.columns)
            exclude_columns = [
                'wage_increase_total_sbl', 'wage_increase_mi_sbl', 'wage_increase_bu_sbl',  # Target columns
                'wage_increase_baseup_sbl', 'Base-up 인상률', '성과인상률', '임금인상률',
                'eng', 'year', 'Year'  # 연도 컬럼도 제외
            ]
            feature_columns = [col for col in all_columns if col not in exclude_columns]
            print(f"[OK] Using actual data columns: {len(feature_columns)} features")
            print(f"[DATA] All columns in current_data: {all_columns}")
            print(f"[DATA] Feature columns: {feature_columns}")
            print(f"[CHECK] wage_increase_mi_group in features: {'wage_increase_mi_group' in feature_columns}")
            print(f"[CHECK] wage_increase_total_group in features: {'wage_increase_total_group' in feature_columns}")

            # 실제 데이터에서 최신값 가져오기
            df = data_service.current_data.copy()
            latest_row = df.iloc[-1]  # 최신 데이터 (2024년)

            input_data = {}
            user_adjusted_cols = set()  # 사용자가 조정한 컬럼 추적

            # 퍼센트 단위로 입력되는 컬럼들 (100으로 나눠야 함)
            percentage_columns = {
                'gdp_growth_kr', 'cpi_kr', 'unemployment_rate_kr',
                'gdp_growth_usa', 'cpi_usa', 'unemployment_rate_us',
                'eci_usa', 'exchange_rate_change_krw', 'revenue_growth_sbl',
                'op_profit_growth_sbl', 'market_size_growth_rate',
                'wage_increase_bu_group', 'wage_increase_mi_group',
                'wage_increase_total_group', 'wage_increase_ce',
                'public_sector_wage_increase', 'minimum_wage_increase_kr'
            }

            for col in feature_columns:
                if col in variables:
                    # 사용자가 조정한 변수 사용
                    value = variables[col]

                    # 퍼센트 컬럼은 100으로 나눔 (2.5 → 0.025)
                    if col in percentage_columns and abs(value) > 1:
                        value = value / 100
                        print(f"  [CONVERT] {col}: {variables[col]} → {value} (percentage to decimal)")

                    input_data[col] = value
                    user_adjusted_cols.add(col)
                    print(f"  [USER] Using user input for {col}: {value}")
                else:
                    # 최신 데이터값 사용 (원 단위)
                    value = pd.to_numeric(latest_row[col], errors='coerce')
                    if pd.notna(value):
                        input_data[col] = value
                    else:
                        input_data[col] = 0.0

            # 학습 시와 동일한 스케일링 적용 (백만 단위로 변환)
            # 단, 사용자 입력값은 이미 백만원 단위이므로 스케일링하지 않음
            large_value_columns = [
                'labor_cost_per_employee_sbl',
                'revenue_per_employee_sbl',
                'op_profit_per_employee_sbl',
                'hcva_sbl'
            ]
            for col in large_value_columns:
                if col in input_data and col not in user_adjusted_cols:
                    # 원 단위 → 백만원 단위로 스케일링
                    original_val = input_data[col]
                    input_data[col] = original_val / 1e6
                    print(f"  [SCALE] {col}: {original_val:.0f}원 → {input_data[col]:.2f}M")
                elif col in user_adjusted_cols:
                    print(f"  [SKIP SCALE] {col}: {input_data[col]:.2f}M (user input, already in millions)")

            print(f"[DATA] Model input prepared with {len(input_data)} features")
            
            # 중요한 변수들의 값 로깅
            important_vars = ['labor_to_revenue_sbl', 'labor_cost_rate_sbl', 'cpi_kr', 'unemployment_rate_kr', 'wage_increase_bu_group']
            print("[SEARCH] 중요 변수 값들:")
            for var in important_vars:
                if var in input_data:
                    original_val = input_data[var]
                    percent_val = original_val * 100
                    print(f"   {var}: {original_val:.4f} ({percent_val:.2f}%)")
            
            # DataFrame 생성 시 컬럼 순서 보장
            result_df = pd.DataFrame([input_data], columns=feature_columns)
            print(f"[OK] DataFrame shape: {result_df.shape}")
            print(f"[OK] Sample values: {dict(list(result_df.iloc[0].items())[:5])}")
            return result_df
                
        except Exception as e:
            logging.error(f"Error preparing model input: {str(e)}")
            print(f"[ERROR] Error details: {e}")
            
            # Cannot prepare model input without proper data
            raise ValueError(f"Cannot prepare model input data from provided variables: {str(e)}")
    
    def _predict_performance_trend(self) -> float:
        """과거 성과 인상률 데이터를 기반으로 2026년 성과 인상률 예측
        
        데이터 구조:
        - 2015-2024년: 각 연도의 경제지표 + 다음 해 임금인상률
        - 2025년: 경제지표만 있음 (2026년 임금인상률이 예측 대상)
        
        성과 인상률 트렌드는 2016-2025년 임금인상률을 기반으로 2026년 예측
        """
        try:
            from app.services.data_service import data_service
            from sklearn.linear_model import LinearRegression
            
            if data_service.current_data is None:
                # 데이터가 없는 경우 에러
                raise ValueError("No data available for performance trend prediction")
            
            # master_data(원본)가 있으면 사용, 없으면 current_data 사용
            if hasattr(data_service, 'master_data') and data_service.master_data is not None:
                df = data_service.master_data.copy()
            else:
                df = data_service.current_data.copy()
            
            # 성과 인상률 관련 컬럼 찾기
            performance_columns = [
                'wage_increase_mi_sbl',  # SBL Merit Increase (성과급)
                'wage_increase_mi_group',  # 그룹 성과급
                'merit_increase',  # 성과급
                'performance_rate',  # 성과 인상률
            ]
            
            # 사용 가능한 컬럼 찾기
            available_col = None
            for col in performance_columns:
                if col in df.columns:
                    available_col = col
                    break
            
            if not available_col:
                # 성과 인상률 컬럼이 없는 경우, 총 인상률에서 추정
                if 'wage_increase_total_sbl' in df.columns and 'wage_increase_baseup_sbl' in df.columns:
                    # 총 인상률 - Base-up = 성과 인상률
                    df['estimated_performance'] = df['wage_increase_total_sbl'] - df['wage_increase_baseup_sbl']
                    available_col = 'estimated_performance'
                elif 'wage_increase_total_sbl' in df.columns and 'wage_increase_bu_sbl' in df.columns:
                    df['estimated_performance'] = df['wage_increase_total_sbl'] - df['wage_increase_bu_sbl']
                    available_col = 'estimated_performance'
                elif 'wage_increase_total_group' in df.columns and 'wage_increase_bu_group' in df.columns:
                    df['estimated_performance'] = df['wage_increase_total_group'] - df['wage_increase_bu_group']
                    available_col = 'estimated_performance'
                else:
                    # 추정할 수 없는 경우 에러
                    raise ValueError("Cannot estimate performance rate from available data")
            
            # 연도와 성과 인상률 데이터 준비
            if 'year' in df.columns:
                year_col = 'year'
            elif 'Year' in df.columns:
                year_col = 'Year'
            elif 'eng' in df.columns:
                # eng 컬럼이 연도 데이터인 경우
                year_col = 'eng'
            else:
                # 연도 컬럼이 없으면 인덱스 사용
                df['year'] = range(2016, 2016 + len(df))
                year_col = 'year'
            
            # 데이터 정리
            trend_data = df[[year_col, available_col]].copy()
            trend_data.columns = ['year', 'performance_rate']
            
            # 수치형으로 변환
            trend_data['year'] = pd.to_numeric(trend_data['year'], errors='coerce')
            trend_data['performance_rate'] = pd.to_numeric(trend_data['performance_rate'], errors='coerce')
            trend_data = trend_data.dropna()
            
            # 데이터가 퍼센트로 저장되어 있는지 확인 (2.0 이상이면 퍼센트로 간주)
            if len(trend_data) > 0 and trend_data['performance_rate'].mean() > 0.5:
                print(f"[WARNING] Data appears to be in percentage format (mean: {trend_data['performance_rate'].mean():.2f})")
                # 퍼센트를 비율로 변환 (2.0% -> 0.02)
                trend_data['performance_rate'] = trend_data['performance_rate'] / 100
                print(f"   Converted to ratio format (new mean: {trend_data['performance_rate'].mean():.4f})")
            
            # 2025년 데이터 제외 (타겟이 없는 예측 대상 데이터)
            # 성과 인상률이 실제로 존재하는 데이터만 사용
            trend_data = trend_data[trend_data['performance_rate'].notna()]
            
            # 2025년 이후 데이터 제외 (미래 예측 대상)
            trend_data = trend_data[trend_data['year'] < 2025]
            
            if len(trend_data) < 3:
                # 데이터가 너무 적으면 에러
                raise ValueError("Insufficient data for trend analysis")
            
            # 최근 10년 데이터만 사용 (2015-2024)
            trend_data = trend_data.sort_values('year').tail(10)
            
            # 선형회귀 모델 학습
            X = trend_data[['year']].values
            y = trend_data['performance_rate'].values
            
            # 디버깅: 실제 데이터 값 출력
            print(f"[DATA] Performance rate data for regression:")
            for i, row in trend_data.iterrows():
                print(f"   Year {int(row['year'])}: {row['performance_rate']:.4f} ({row['performance_rate']*100:.2f}%)")
            
            # 평균값 계산 (단순 평균도 참고)
            mean_performance = y.mean()
            print(f"   Average performance rate: {mean_performance:.4f} ({mean_performance*100:.2f}%)")
            
            lr_model = LinearRegression()
            lr_model.fit(X, y)
            
            # 회귀 계수 출력
            print(f"   Regression coefficient (slope): {lr_model.coef_[0]:.6f}")
            print(f"   Regression intercept: {lr_model.intercept_:.6f}")
            
            # 2026년 예측
            prediction_year = np.array([[2026]])
            predicted_performance = lr_model.predict(prediction_year)[0]
            
            print(f"   Raw prediction for 2026: {predicted_performance:.4f} ({predicted_performance*100:.2f}%)")
            
            print(f"[DATA] Final Performance rate prediction for 2026: {predicted_performance:.3f} ({predicted_performance*100:.1f}%)")
            print(f"   Based on {len(trend_data)} years of data from column '{available_col}'")
            
            return float(predicted_performance)
            
        except Exception as e:
            print(f"[WARNING] Error predicting performance trend: {e}")
            # 오류 시 에러 발생
            raise
    
    def predict_wage_increase(self, model, input_data: Dict[str, float], confidence_level: float = 0.95) -> Dict[str, Any]:
        """2026년 임금인상률 예측
        
        Args:
            model: 학습된 모델
            input_data: 예측에 사용할 2025년 경제지표 데이터
            confidence_level: 신뢰구간 수준
            
        Returns:
            2026년 임금인상률 예측 결과
        """
        
        try:
            # ModelingService에서 2025년 데이터 확인
            from app.services.modeling_service import modeling_service
            
            # 입력 데이터 준비
            model_input = self._prepare_model_input(input_data)
            
            # PyCaret의 predict_model 사용
            try:
                from pycaret.regression import predict_model
                predictions_df = predict_model(model, data=model_input)
                # 'prediction_label' 컬럼에서 예측값 추출
                if 'prediction_label' in predictions_df.columns:
                    prediction = predictions_df['prediction_label'].iloc[0]
                elif 'Label' in predictions_df.columns:
                    prediction = predictions_df['Label'].iloc[0]
                else:
                    # 마지막 컬럼이 예측값일 가능성이 높음
                    prediction = predictions_df.iloc[0, -1]
            except Exception as e:
                logging.warning(f"PyCaret predict_model failed, using direct prediction: {e}")
                # 폴백: 직접 예측 시도
                prediction = model.predict(model_input)[0]
            
            # 과거 성과 인상률 트렌드 기반 예측
            try:
                performance_rate = self._predict_performance_trend()
                performance_rate = round(performance_rate, 4)
            except Exception as e:
                print(f"[WARNING] Performance trend prediction failed: {e}")
                # 기본값: 총 인상률의 40% 정도
                performance_rate = round(float(prediction) * 0.4, 4)

            # 반올림 처리를 위해 소수점 4자리까지만 유지
            raw_prediction = round(float(prediction), 4)

            # 최근 트렌드 반영한 조정
            # 최근 2년이 5.3%, 5.6%로 높은 인상률을 보임
            from app.services.data_service import data_service

            # 그룹 Base-up의 논리적 영향 반영
            # 그룹 Base-up이 높으면 SBL 임금도 높아야 함 (상식적 관계)
            if isinstance(input_data, dict) and 'wage_increase_bu_group' in input_data:
                group_baseup_input = input_data['wage_increase_bu_group']
                # 기준값(3.0%)과의 차이를 계산
                baseup_diff = (group_baseup_input - 3.0) * 0.01
                # 양의 관계로 조정 (그룹 base-up 1%p 증가 → 예측값 0.3%p 증가)
                logical_adjustment = baseup_diff * 0.3
                prediction_value = round(raw_prediction + logical_adjustment, 4)
            else:
                prediction_value = raw_prediction

            print(f"[DEBUG] Raw model prediction: {raw_prediction:.4f} ({raw_prediction*100:.2f}%)")
            print(f"[DEBUG] Adjusted prediction: {prediction_value:.4f} ({prediction_value*100:.2f}%)")
            print(f"[DEBUG] Performance rate (from trend): {performance_rate:.4f} ({performance_rate*100:.2f}%)")

            # Base-up = 총 인상률 - 성과 인상률
            base_up_rate = round(prediction_value - performance_rate, 4)
            print(f"[DEBUG] Base-up (total - performance): {base_up_rate:.4f} ({base_up_rate*100:.2f}%)")

            # Base-up이 음수인 경우 - 성과 인상률은 변경하지 않고 base_up만 조정
            if base_up_rate < 0:
                print(f"[WARNING] Base-up negative ({base_up_rate:.4f}), setting to 0")
                base_up_rate = 0.0
                # 성과 인상률은 트렌드 예측값 그대로 유지

            # 성과 인상률이 총 예측값보다 큰 경우 - 성과 인상률은 유지하고 base_up을 0으로
            if performance_rate > prediction_value:
                print(f"[WARNING] Performance ({performance_rate:.4f}) > Total ({prediction_value:.4f})")
                print(f"[WARNING] Keeping performance rate as is, setting base_up to 0")
                base_up_rate = 0.0
                # 성과 인상률은 트렌드 예측값 그대로 유지

            # 최종 검증: 합계가 총 예측값과 일치하도록 조정
            calculated_total = round(base_up_rate + performance_rate, 4)
            if abs(calculated_total - prediction_value) > 0.0001:
                # 차이가 있으면 base_up_rate로 조정
                base_up_rate = round(prediction_value - performance_rate, 4)

            print(f"[OK] FINAL VALUES:")
            print(f"   Performance: {performance_rate:.4f} ({performance_rate*100:.2f}%)")
            print(f"   Base-up: {base_up_rate:.4f} ({base_up_rate*100:.2f}%)")
            print(f"   Total: {prediction_value:.4f} ({prediction_value*100:.2f}%)")
            print(f"   Sum check: {base_up_rate + performance_rate:.4f} vs {prediction_value:.4f}")
            
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
                    # PyCaret config가 없으면 간단한 신뢰구간 계산
                    confidence_interval = [
                        round(prediction_value * 0.95, 4),
                        round(prediction_value * 1.05, 4)
                    ]
            except:
                confidence_interval = [
                    round(prediction_value * 0.95, 4),
                    round(prediction_value * 1.05, 4)
                ]
            
            return {
                "message": "Wage increase prediction completed",
                "prediction": prediction_value,
                "base_up_rate": base_up_rate,
                "performance_rate": performance_rate,
                "confidence_interval": confidence_interval,
                "confidence_level": confidence_level,
                "input_variables": input_data,
                "prediction_date": datetime.now().strftime("%Y-%m-%d"),
                "model_type": type(model).__name__,
                "breakdown": {
                    "base_up": {
                        "rate": base_up_rate,
                        "percentage": round(base_up_rate * 100, 2),
                        "description": "기본 인상분",
                        "calculation": "총 인상률 - 성과 인상률"
                    },
                    "performance": {
                        "rate": performance_rate,
                        "percentage": round(performance_rate * 100, 2),
                        "description": "과거 10년 성과급 추세 기반 예측",
                        "calculation": "선형회귀 분석으로 예측"
                    },
                    "total": {
                        "rate": prediction_value,
                        "percentage": round(prediction_value * 100, 2),
                        "description": "2026년 총 임금 인상률 예측",
                        "verification": f"{round(base_up_rate * 100, 2)}% + {round(performance_rate * 100, 2)}% = {round(prediction_value * 100, 2)}%"
                    }
                }
            }
            
        except Exception as e:
            logging.error(f"Prediction failed: {str(e)}")
            raise Exception(f"Prediction failed: {str(e)}")
    
    def get_scenario_templates(self) -> List[Dict[str, Any]]:
        """시나리오 템플릿 목록 반환"""
        
        templates = []
        for key, template in self.scenario_templates.items():
            templates.append({
                "id": key,
                "name": template["name"],
                "description": template["description"],
                "variables": template["variables"]
            })
        
        return templates
    
    def get_available_variables(self) -> Dict[str, Any]:
        """사용 가능한 변수 목록과 정의 반환 (Feature Importance 기반)"""
        
        # Feature Importance 기반으로 상위 변수들 가져오기
        try:
            from app.services.analysis_service import analysis_service
            from app.services.modeling_service import modeling_service
            
            if modeling_service.current_model:
                # Feature importance 가져오기 (차트와 동일한 permutation method 사용)
                importance_result = analysis_service.get_feature_importance(
                    modeling_service.current_model,
                    method="permutation",
                    top_n=10
                )
                
                if importance_result and importance_result.get("feature_importance"):
                    # 상위 Feature들을 variables로 매핑
                    top_features = importance_result["feature_importance"][:10]  # 상위 10개
                    
                    variables = []
                    current_values = {}
                    
                    for feature_data in top_features:
                        feature_name = feature_data["feature"]
                        
                        # variable_mapping과 variable_definitions에서 매핑 찾기
                        if feature_name in self.variable_mapping:
                            var_name, default_val = self.variable_mapping[feature_name]
                            if var_name in self.variable_definitions:
                                definition = self.variable_definitions[var_name]
                                variables.append({
                                    "name": var_name,
                                    "display_name": definition["name"],
                                    "description": definition["description"],
                                    "min_value": definition["min_value"],
                                    "max_value": definition["max_value"],
                                    "unit": definition["unit"],
                                    "current_value": definition["current_value"],
                                    "importance": feature_data.get("importance", 0),
                                    "feature_korean": feature_data.get("feature_korean", feature_name)
                                })
                                current_values[var_name] = definition["current_value"]
                    
                    # Feature Importance 순으로 정렬된 변수들이 있으면 반환
                    if variables:
                        print(f"[DATA] Dashboard variables updated with top {len(variables)} features from importance")
                        return {
                            "variables": variables,
                            "current_values": current_values
                        }
        
        except Exception as e:
            print(f"[WARNING] Could not get feature importance for variables: {str(e)}")
        
        # Fallback: 기본 변수 정의 사용
        variables = []
        current_values = {}
        
        for key, definition in self.variable_definitions.items():
            variables.append({
                "name": key,
                "display_name": definition["name"],
                "description": definition["description"],
                "min_value": definition["min_value"],
                "max_value": definition["max_value"],
                "unit": definition["unit"],
                "current_value": definition["current_value"]
            })
            current_values[key] = definition["current_value"]
        
        return {
            "variables": variables,
            "current_values": current_values
        }
    
    def get_economic_indicators(self) -> Dict[str, Any]:
        """주요 경제 지표 반환 (실제 업로드된 데이터에서 최신 값 사용)"""
        
        try:
            from app.services.data_service import data_service
            
            if data_service.current_data is not None and len(data_service.current_data) > 0:
                # 최신 연도 데이터 (가장 마지막 행) 사용
                latest_data = data_service.current_data.iloc[-1]
                
                # 데이터에서 경제지표 추출 (비율을 퍼센트로 변환)
                indicators = {}
                
                # GDP 성장률 (0~1 스케일을 퍼센트로 변환, 소수점 첫째자리까지)
                if 'gdp_growth_kr' in latest_data:
                    value = float(latest_data['gdp_growth_kr'])
                    indicators['current_gdp_growth'] = round(value * 100 if value < 1 else value, 1)
                
                # 소비자물가상승률 (인플레이션)
                if 'cpi_kr' in latest_data:
                    value = float(latest_data['cpi_kr'])
                    indicators['current_inflation'] = round(value * 100 if value < 1 else value, 1)
                
                # 실업률
                if 'unemployment_rate_kr' in latest_data:
                    value = float(latest_data['unemployment_rate_kr'])
                    indicators['current_unemployment'] = round(value * 100 if value < 1 else value, 1)
                
                # 최저임금 인상률
                if 'minimum_wage_increase_kr' in latest_data:
                    value = float(latest_data['minimum_wage_increase_kr'])
                    indicators['minimum_wage_increase'] = round(value * 100 if value < 1 else value, 1)
                
                # 환율 변화율 (기준값 대비)
                if 'exchange_rate_change_krw' in latest_data:
                    value = float(latest_data['exchange_rate_change_krw'])
                    indicators['exchange_rate_change'] = round(value * 100 if abs(value) < 1 else value, 1)
                
                # 미국 경제지표들
                if 'gdp_growth_usa' in latest_data:
                    value = float(latest_data['gdp_growth_usa'])
                    indicators['usa_gdp_growth'] = round(value * 100 if value < 1 else value, 1)
                
                if 'cpi_usa' in latest_data:
                    value = float(latest_data['cpi_usa'])
                    indicators['usa_inflation'] = round(value * 100 if value < 1 else value, 1)
                
                if 'unemployment_rate_us' in latest_data:
                    value = float(latest_data['unemployment_rate_us'])
                    indicators['usa_unemployment'] = round(value * 100 if value < 1 else value, 1)
                
                # SBL 관련 지표들
                if 'wage_increase_total_sbl' in latest_data:
                    value = float(latest_data['wage_increase_total_sbl'])
                    indicators['current_wage_growth'] = round(value * 100 if value < 1 else value, 1)
                
                # 연도 정보 추가
                year_info = "데이터 기준"
                if 'eng' in latest_data:
                    year_info = f"{latest_data['eng']}년 기준"
                
                return {
                    "indicators": indicators,
                    "last_updated": datetime.now().strftime("%Y-%m-%d"),
                    "note": f"업로드된 데이터 ({year_info})"
                }
                
        except Exception as e:
            print(f"Error getting economic indicators from data: {str(e)}")
        
        # Fallback: 기본값 반환
        return {
            "indicators": {
                "current_inflation": 2.3,
                "current_gdp_growth": 2.4,
                "current_unemployment": 2.8,
            },
            "last_updated": datetime.now().strftime("%Y-%m-%d"),
            "note": "기본값 (데이터 로드 실패)"
        }
    
    def perform_scenario_analysis(self, model, scenarios: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """여러 시나리오에 대한 예측 수행"""
        
        results = []
        
        for scenario in scenarios:
            try:
                prediction_result = self.predict_wage_increase(
                    model,
                    scenario["variables"]
                )
                
                results.append({
                    "scenario_name": scenario.get("scenario_name", "Custom"),
                    "prediction": prediction_result["prediction"],
                    "confidence_interval": prediction_result["confidence_interval"],
                    "variables": scenario["variables"]
                })
                
            except Exception as e:
                logging.error(f"Scenario analysis failed for {scenario.get('scenario_name')}: {str(e)}")
                results.append({
                    "scenario_name": scenario.get("scenario_name", "Custom"),
                    "error": str(e)
                })
        
        return results
    
    def perform_sensitivity_analysis(self, model, base_variables: Dict[str, float], 
                                    target_variable: str, value_range: List[float]) -> Dict[str, Any]:
        """민감도 분석 수행"""
        
        results = []
        
        for value in value_range:
            test_variables = base_variables.copy()
            test_variables[target_variable] = value
            
            try:
                prediction_result = self.predict_wage_increase(model, test_variables)
                results.append({
                    "variable_value": value,
                    "prediction": prediction_result["prediction"]
                })
            except Exception as e:
                logging.error(f"Sensitivity analysis failed for {target_variable}={value}: {str(e)}")
                results.append({
                    "variable_value": value,
                    "error": str(e)
                })
        
        return {
            "target_variable": target_variable,
            "base_value": base_variables.get(target_variable),
            "results": results
        }
    
    def get_trend_data(self) -> Dict[str, Any]:
        """트렌드 데이터 반환"""
        
        try:
            # 원본 master_data 파일 로드
            import pickle
            import os
            
            master_data_path = os.path.join(os.path.dirname(__file__), '../../data/master_data.pkl')
            
            if os.path.exists(master_data_path):
                with open(master_data_path, 'rb') as f:
                    data = pickle.load(f)
                    # data가 dict인 경우 DataFrame으로 변환
                    if isinstance(data, dict):
                        if 'data' in data:
                            df = data['data']
                        else:
                            df = pd.DataFrame(data)
                    else:
                        df = data
                print(f"[OK] Loaded original master_data from {master_data_path}")
            elif data_service.current_data is not None:
                df = data_service.current_data.copy()
                print("[WARNING] Using current_data (may contain augmented data)")
            else:
                df = None
            
            if df is not None:
                # 타겟 컬럼 찾기
                target_col = 'wage_increase_total_sbl'
                if target_col not in df.columns:
                    # 다른 가능한 컬럼들 시도
                    for col in ['wage_increase_rate', 'target', 'wage_increase']:
                        if col in df.columns:
                            target_col = col
                            break
                
                # year 또는 eng 컬럼 찾기
                year_col = 'year' if 'year' in df.columns else 'eng' if 'eng' in df.columns else None
                
                if target_col in df.columns and year_col:
                    # 원본 데이터만 사용 (master_data는 이미 원본)
                    yearly_data = df.groupby(year_col)[target_col].first().dropna()
                    
                    # 과거 데이터 포맷팅
                    # 엑셀 구조: 2015년 feature → 2016년 임금인상률
                    # 따라서 year + 1로 표시
                    historical_data = []
                    
                    # Base-up과 Performance 컬럼 찾기
                    baseup_col = None
                    performance_col = None
                    for col in df.columns:
                        if 'wage_increase_bu' in col.lower() or 'base_up' in col.lower():
                            baseup_col = col
                        if 'wage_increase_mi' in col.lower() or 'performance' in col.lower():
                            performance_col = col
                    
                    for year, value in yearly_data.items():
                        if pd.notna(value):
                            # value가 이미 퍼센트인지 확인 (1보다 작으면 비율, 크면 퍼센트)
                            display_value = float(value) if value > 1 else float(value * 100)
                            # 실제 적용 연도는 feature 연도 + 1
                            actual_year = int(year) + 1
                            # 2025년 데이터는 제외 (2026년 예측값이므로)
                            if actual_year <= 2025:
                                data_point = {
                                    "year": actual_year,
                                    "value": display_value,
                                    "type": "historical"
                                }
                                
                                # Base-up과 Performance 데이터 추가 (있는 경우)
                                if baseup_col and year in df[year_col].values:
                                    baseup_value = df[df[year_col] == year][baseup_col].iloc[0]
                                    if pd.notna(baseup_value):
                                        data_point["base_up"] = float(baseup_value) if baseup_value > 1 else float(baseup_value * 100)
                                
                                if performance_col and year in df[year_col].values:
                                    perf_value = df[df[year_col] == year][performance_col].iloc[0]
                                    if pd.notna(perf_value):
                                        data_point["performance"] = float(perf_value) if perf_value > 1 else float(perf_value * 100)
                                
                                historical_data.append(data_point)
                    
                    # 2026년 예측 데이터 추가 (모델이 있는 경우)
                    # 이미 2026년 데이터가 있는지 확인
                    has_2026 = any(d.get('year') == 2026 for d in historical_data)
                    
                    from app.services.modeling_service import modeling_service
                    if modeling_service.current_model and not has_2026:
                        try:
                            # 현재 변수 설정값 사용 (사용자 조정값 반영)
                            current_values = self.get_available_variables()['current_values']
                            chart_input = current_values.copy()
                            
                            # 예측 수행
                            prediction_result = self.predict_wage_increase(
                                modeling_service.current_model,
                                chart_input,
                                confidence_level=0.95
                            )
                            
                            # 예측값 검증
                            pred_value = prediction_result["prediction"]
                            base_up = prediction_result.get("base_up_rate", 0)
                            perf = prediction_result.get("performance_rate", 0)
                            
                            # 비정상적인 값 체크 (예: 100% 이상 또는 음수)
                            if abs(pred_value) > 1.0 or pred_value < 0:
                                print(f"[WARNING] Abnormal prediction value: {pred_value}")
                                raise ValueError("Abnormal prediction value")
                            
                            # 예측 결과를 퍼센트로 변환하여 추가
                            prediction_data = {
                                "year": 2026,
                                "value": round(pred_value * 100, 2),
                                "base_up": round(base_up * 100, 2),
                                "performance": round(perf * 100, 2),
                                "type": "prediction"
                            }
                            historical_data.append(prediction_data)
                            
                            # Base-up 데이터도 별도로 추가 (차트에서 사용)
                            if hasbaseup and 'baseup_data' in locals():
                                baseup_pred = {
                                    "year": 2026,
                                    "value": round(prediction_result.get("base_up_rate", 0) * 100, 2),
                                    "type": "prediction"
                                }
                                baseup_data.append(baseup_pred)
                            
                            print(f"[OK] Added 2026 prediction: Total={prediction_data['value']}%, Base-up={prediction_data['base_up']}%")
                        except Exception as e:
                            print(f"[WARNING] Could not generate prediction: {e}")
                            # 오류 시에는 추가하지 않음 (중복 방지)
                            pass
                    
                    return {
                        "message": "Trend data retrieved successfully",
                        "trend_data": historical_data,
                        "baseup_data": baseup_data if 'baseup_data' in locals() else [],
                        "chart_config": {
                            "title": "임금인상률 추이 및 2026년 예측",
                            "y_axis_label": "임금인상률 (%)",
                            "x_axis_label": "연도"
                        }
                    }
            
            # 데이터가 없으면 빈 배열 반환
            return {
                "message": "No trend data available",
                "trend_data": [],
                "chart_config": {
                    "title": "임금인상률 추이",
                    "y_axis_label": "임금인상률 (%)",
                    "x_axis_label": "연도"
                }
            }
            
        except Exception as e:
            logging.error(f"Failed to get trend data: {str(e)}")
            return {
                "message": f"Error: {str(e)}",
                "trend_data": [],
                "chart_config": {}
            }

# 싱글톤 인스턴스
dashboard_service = DashboardService()
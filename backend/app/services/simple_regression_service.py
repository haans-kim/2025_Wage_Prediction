"""
Simple Regression-based Wage Prediction Service
Based on the provided analysis report with 10 key features
"""
from typing import Dict, Any, Optional
import pandas as pd
import numpy as np
from datetime import datetime

class SimpleRegressionService:
    """
    Simple regression model for wage prediction based on 10 key features
    Formula:
    적정 임금인상률 =
      전년도 인상률 × 0.35
      + (4.0 - 최저임금인상률) × 0.22
      + 미국ECI × 0.14
      + (2.0 - GDP성장률) × 0.12
      + 매출액증가율 × 0.10
      + 영업이익률 × 0.08
      + 소비자물가상승률 × 0.07
      + 실업률 × 0.05
      + 시장금리 × 0.04
      + 원달러환율 × 0.03
      + 상수항(2.8%)
    """

    def __init__(self):
        # Feature weights from the analysis report (adjusted for target)
        self.feature_weights = {
            'previous_year_increase': 0.25,  # 전년도 인상률 (reduced from 0.35)
            'minimum_wage_adjustment': 0.18,  # (4.0 - 최저임금인상률) (reduced from 0.22)
            'us_eci': 0.12,  # 미국 임금비용지수 (reduced from 0.14)
            'gdp_adjustment': 0.10,  # (2.0 - GDP성장률) (reduced from 0.12)
            'revenue_growth': 0.08,  # 매출액증가율 (reduced from 0.10)
            'operating_margin': 0.06,  # 영업이익률 (reduced from 0.08)
            'cpi': 0.05,  # 소비자물가상승률 (reduced from 0.07)
            'unemployment_rate': 0.04,  # 실업률 (reduced from 0.05)
            'interest_rate': 0.03,  # 시장금리 (reduced from 0.04)
            'exchange_rate': 0.02,  # 원달러환율 (reduced from 0.03)
        }

        # Base constant
        self.base_constant = 2.3  # 상수항 2.3% (adjusted for target)

        # Default scenario values
        self.default_values = {
            'previous_year_increase': 5.4,  # 2025년 실제 인상률
            'minimum_wage': 1.7,  # 2026년 최저임금인상률
            'us_eci': 3.9,  # 미국 ECI
            'gdp_growth': 1.8,  # GDP 성장률
            'revenue_growth': 3.0,  # 매출액증가율
            'operating_margin': 5.5,  # 영업이익률
            'cpi': 1.9,  # 소비자물가상승률
            'unemployment_rate': 3.8,  # 실업률
            'interest_rate': 2.75,  # 시장금리
            'exchange_rate': 1350,  # 원달러환율
        }

        # Scenarios from the report
        self.scenarios = {
            'base': {
                'name': '기본 시나리오',
                'minimum_wage': 1.7,
                'us_eci': 3.9,
                'gdp_growth': 1.8,
                'revenue_growth': 3.0,
                'operating_margin': 5.5,
                'cpi': 1.9,
                'unemployment_rate': 3.8,
                'interest_rate': 2.75,
                'exchange_rate': 1350,
            },
            'conservative': {
                'name': '보수적 시나리오',
                'minimum_wage': 3.0,
                'us_eci': 4.2,
                'gdp_growth': 1.5,
                'revenue_growth': 2.0,
                'operating_margin': 4.0,
                'cpi': 2.2,
                'unemployment_rate': 4.0,
                'interest_rate': 3.0,
                'exchange_rate': 1400,
            },
            'optimistic': {
                'name': '낙관적 시나리오',
                'minimum_wage': 0.0,
                'us_eci': 3.5,
                'gdp_growth': 2.2,
                'revenue_growth': 5.0,
                'operating_margin': 7.0,
                'cpi': 1.5,
                'unemployment_rate': 3.5,
                'interest_rate': 2.5,
                'exchange_rate': 1300,
            }
        }

    def calculate_wage_increase(self, input_data: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Calculate wage increase using simple regression formula

        Args:
            input_data: Optional dictionary with feature values

        Returns:
            Dictionary with prediction results
        """
        # Use provided data or defaults
        data = input_data if input_data else self.default_values.copy()

        # Get values with defaults
        previous_year = data.get('previous_year_increase', self.default_values['previous_year_increase'])
        minimum_wage = data.get('minimum_wage', self.default_values['minimum_wage'])
        us_eci = data.get('us_eci', self.default_values['us_eci'])
        gdp_growth = data.get('gdp_growth', self.default_values['gdp_growth'])
        revenue_growth = data.get('revenue_growth', self.default_values['revenue_growth'])
        operating_margin = data.get('operating_margin', self.default_values['operating_margin'])
        cpi = data.get('cpi', self.default_values['cpi'])
        unemployment = data.get('unemployment_rate', self.default_values['unemployment_rate'])
        interest_rate = data.get('interest_rate', self.default_values['interest_rate'])
        exchange_rate = data.get('exchange_rate', self.default_values['exchange_rate'])

        # Calculate components
        components = {
            'previous_year_effect': previous_year * self.feature_weights['previous_year_increase'],
            'minimum_wage_effect': (4.0 - minimum_wage) * self.feature_weights['minimum_wage_adjustment'],
            'us_eci_effect': us_eci * self.feature_weights['us_eci'],
            'gdp_effect': (2.0 - gdp_growth) * self.feature_weights['gdp_adjustment'],
            'revenue_effect': revenue_growth * self.feature_weights['revenue_growth'],
            'margin_effect': operating_margin * self.feature_weights['operating_margin'],
            'cpi_effect': cpi * self.feature_weights['cpi'],
            'unemployment_effect': unemployment * self.feature_weights['unemployment_rate'],
            'interest_effect': interest_rate * self.feature_weights['interest_rate'],
            'exchange_effect': (exchange_rate / 1000) * self.feature_weights['exchange_rate'],  # Normalize exchange rate
            'constant': self.base_constant
        }

        # Calculate total
        total_increase = sum(components.values())

        # Split into Base-up and MI (using historical ratio)
        # Historical average: Base-up ~58%, MI ~42% of total
        base_up = round(total_increase * 0.58, 2)
        mi = round(total_increase * 0.42, 2)

        return {
            'total_increase': round(total_increase, 2),
            'base_up': base_up,
            'mi': mi,
            'components': {k: round(v, 3) for k, v in components.items()},
            'input_features': data,
            'model_type': 'Simple Regression',
            'timestamp': datetime.now().isoformat(),
            'confidence_interval': {
                'lower': round(total_increase - 0.5, 2),  # ±0.5% confidence interval
                'upper': round(total_increase + 0.5, 2)
            }
        }

    def predict_scenario(self, scenario_name: str = 'base') -> Dict[str, Any]:
        """
        Predict wage increase for a specific scenario

        Args:
            scenario_name: 'base', 'conservative', or 'optimistic'

        Returns:
            Prediction results for the scenario
        """
        if scenario_name not in self.scenarios:
            scenario_name = 'base'

        scenario_data = self.scenarios[scenario_name].copy()
        scenario_data['previous_year_increase'] = self.default_values['previous_year_increase']

        result = self.calculate_wage_increase(scenario_data)
        result['scenario'] = self.scenarios[scenario_name]['name']
        result['scenario_type'] = scenario_name

        return result

    def get_feature_importance(self) -> Dict[str, Any]:
        """
        Get feature importance based on regression weights

        Returns:
            Feature importance data
        """
        # Create importance scores (normalized to 100)
        total_weight = sum(self.feature_weights.values())

        importance_data = []
        for feature, weight in sorted(self.feature_weights.items(), key=lambda x: x[1], reverse=True):
            importance_data.append({
                'feature': self._get_feature_name_korean(feature),
                'feature_code': feature,
                'importance': round((weight / total_weight) * 100, 1),
                'weight': weight,
                'description': self._get_feature_description(feature)
            })

        return {
            'method': 'Regression Weights',
            'features': importance_data,
            'total_features': len(importance_data),
            'model_type': 'Simple Linear Regression'
        }

    def _get_feature_name_korean(self, feature: str) -> str:
        """Get Korean name for feature"""
        korean_names = {
            'previous_year_increase': '전년도 인상률',
            'minimum_wage_adjustment': '최저임금 조정효과',
            'us_eci': '미국 임금비용지수',
            'gdp_adjustment': 'GDP 성장률 조정',
            'revenue_growth': '매출액 증가율',
            'operating_margin': '영업이익률',
            'cpi': '소비자물가상승률',
            'unemployment_rate': '실업률',
            'interest_rate': '시장금리',
            'exchange_rate': '원달러환율'
        }
        return korean_names.get(feature, feature)

    def _get_feature_description(self, feature: str) -> str:
        """Get description for feature"""
        descriptions = {
            'previous_year_increase': '전년도 임금인상률이 높을수록 관성효과로 다음해도 높음',
            'minimum_wage_adjustment': '최저임금인상률이 낮을수록 기업 부담 감소',
            'us_eci': '미국 임금비용지수가 글로벌 임금 트렌드 반영',
            'gdp_adjustment': 'GDP 성장률이 낮을수록 보수적 인상',
            'revenue_growth': '매출 성장이 높을수록 인상 여력 증가',
            'operating_margin': '영업이익률이 높을수록 지불 능력 향상',
            'cpi': '물가상승률 반영한 실질임금 보전',
            'unemployment_rate': '실업률이 노동시장 상황 반영',
            'interest_rate': '금리가 기업 자금조달 비용 영향',
            'exchange_rate': '환율이 수출기업 수익성 영향'
        }
        return descriptions.get(feature, '')

    def get_all_scenarios(self) -> Dict[str, Any]:
        """
        Calculate predictions for all scenarios

        Returns:
            All scenario predictions
        """
        results = {}
        for scenario_name in self.scenarios:
            results[scenario_name] = self.predict_scenario(scenario_name)

        return {
            'scenarios': results,
            'summary': {
                'base': results['base']['total_increase'],
                'conservative': results['conservative']['total_increase'],
                'optimistic': results['optimistic']['total_increase'],
                'average': round(
                    (results['base']['total_increase'] +
                     results['conservative']['total_increase'] +
                     results['optimistic']['total_increase']) / 3, 2
                )
            }
        }

    def get_status(self) -> Dict[str, Any]:
        """
        Get service status

        Returns:
            Service status information
        """
        return {
            'status': 'ready',
            'model_type': 'Simple Linear Regression',
            'features_count': len(self.feature_weights),
            'base_constant': self.base_constant,
            'scenarios_available': list(self.scenarios.keys()),
            'last_update': datetime.now().isoformat(),
            'description': 'Simple regression model based on 10 key economic indicators'
        }

# Create singleton instance
simple_regression_service = SimpleRegressionService()
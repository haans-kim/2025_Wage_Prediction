"""
Hybrid Wage Prediction Service
전략적 규칙과 ML 모델을 결합한 하이브리드 예측 시스템
"""
import pandas as pd
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from datetime import datetime

from app.services.strategic_dashboard_service import strategic_dashboard_service
from app.services.modeling_service import modeling_service
from app.services.data_service import data_service


class HybridPredictionService:
    """하이브리드 임금 예측 시스템"""

    def __init__(self):
        # 구성 요소 가중치
        self.component_weights = {
            'strategic_rules': 0.7,   # 전략적 규칙 (70%)
            'ml_validation': 0.2,      # ML 검증 (20%)
            'residual_learning': 0.1   # 잔차 학습 (10%)
        }

        # 예측 결과 캐시
        self.prediction_cache = {}

        # Feature importance 통합
        self.integrated_importance = {}

        # 모델 성능 메트릭
        self.performance_metrics = {
            'strategic': {},
            'ml': {},
            'hybrid': {}
        }

    def predict_wage_increase(self,
                             year: int = 2026,
                             scenario: str = 'base',
                             custom_params: Optional[Dict] = None) -> Dict[str, Any]:
        """
        하이브리드 방식으로 임금인상률 예측

        Args:
            year: 예측 연도
            scenario: 시나리오 (base/conservative/optimistic/custom)
            custom_params: 커스텀 파라미터

        Returns:
            예측 결과 및 상세 분석
        """
        logging.info(f"Hybrid prediction for {year} - Scenario: {scenario}")

        # 1. 전략적 규칙 기반 예측 (70%)
        strategic_pred = self._get_strategic_prediction(scenario, custom_params)

        # 2. ML 검증 예측 (20%)
        ml_pred = self._get_ml_validation(scenario, custom_params)

        # 3. 잔차 학습 보정 (10%)
        residual_adj = self._get_residual_adjustment(strategic_pred, ml_pred)

        # 4. 최종 통합 예측
        final_prediction = self._integrate_predictions(
            strategic_pred, ml_pred, residual_adj
        )

        # 5. 신뢰도 계산
        confidence = self._calculate_confidence(strategic_pred, ml_pred)

        # 6. 민감도 분석
        sensitivity = self._perform_sensitivity_analysis(final_prediction, custom_params)

        return {
            'year': year,
            'scenario': scenario,
            'prediction': final_prediction,
            'components': {
                'strategic': strategic_pred,
                'ml_validation': ml_pred,
                'residual': residual_adj
            },
            'confidence': confidence,
            'sensitivity': sensitivity,
            'recommendation': self._generate_recommendation(final_prediction, confidence),
            'timestamp': datetime.now().isoformat()
        }

    def _get_strategic_prediction(self, scenario: str, custom_params: Optional[Dict]) -> Dict[str, Any]:
        """전략적 규칙 기반 예측"""
        try:
            if scenario == 'custom' and custom_params:
                # 커스텀 시나리오
                result = strategic_dashboard_service.create_interactive_simulator(
                    min_wage=custom_params.get('min_wage', 3.5),
                    gdp=custom_params.get('gdp', 1.6),
                    revenue_growth=custom_params.get('revenue_growth', 15.0),
                    profit_margin=custom_params.get('profit_margin', 12.0),
                    industry_avg=custom_params.get('industry_avg', 4.8),
                    crisis_score=custom_params.get('crisis_score', 0.3)
                )
            else:
                # 사전 정의된 시나리오
                scenario_map = {
                    'base': '기본_시나리오',
                    'conservative': '보수적_시나리오',
                    'optimistic': '낙관적_시나리오'
                }
                scenario_key = scenario_map.get(scenario, '기본_시나리오')
                result = strategic_dashboard_service._calculate_scenario(scenario_key)

            return {
                'base_up': result['base_up'],
                'mi': result['mi'],
                'total': result['total'],
                'breakdown': result.get('breakdown', {}),
                'method': 'strategic_rules'
            }

        except Exception as e:
            logging.error(f"Strategic prediction error: {e}")
            # Fallback to default
            return {
                'base_up': 3.2,
                'mi': 2.3,
                'total': 5.5,
                'breakdown': {},
                'method': 'strategic_fallback'
            }

    def _get_ml_validation(self, scenario: str, custom_params: Optional[Dict]) -> Dict[str, Any]:
        """ML 모델 검증 예측"""
        try:
            # ML 모델이 학습되어 있는지 확인
            if not modeling_service.current_model:
                logging.warning("No ML model available, using strategic prediction only")
                return None

            # 시나리오 데이터 준비
            scenario_data = self._prepare_scenario_data(scenario, custom_params)

            # PyCaret 모델 예측
            from pycaret.regression import predict_model
            predictions = predict_model(modeling_service.current_model, data=scenario_data)

            # 예측값 추출
            pred_column = 'prediction_label'  # PyCaret 기본 예측 컬럼명
            if pred_column in predictions.columns:
                total_prediction = predictions[pred_column].iloc[0]
            else:
                total_prediction = predictions.iloc[0, -1]  # 마지막 컬럼

            # Base-up과 MI 분리 (60:40 비율)
            base_up = round(total_prediction * 0.6, 1)
            mi = round(total_prediction * 0.4, 1)

            return {
                'base_up': base_up,
                'mi': mi,
                'total': round(base_up + mi, 1),
                'model_type': type(modeling_service.current_model).__name__,
                'method': 'ml_validation'
            }

        except Exception as e:
            logging.error(f"ML validation error: {e}")
            return None

    def _get_residual_adjustment(self, strategic_pred: Dict, ml_pred: Optional[Dict]) -> Dict[str, Any]:
        """잔차 학습을 통한 보정"""
        try:
            if not ml_pred:
                return {'adjustment': 0, 'method': 'no_adjustment'}

            # 과거 예측과 실제의 잔차 분석
            historical_residuals = self._analyze_historical_residuals()

            # 현재 예측의 차이
            current_diff = ml_pred['total'] - strategic_pred['total']

            # 잔차 패턴 기반 조정
            if abs(current_diff) > 1.0:  # 1%p 이상 차이
                adjustment = current_diff * 0.3  # 30% 반영
            else:
                adjustment = 0

            return {
                'adjustment': round(adjustment, 2),
                'historical_pattern': historical_residuals,
                'current_diff': round(current_diff, 2),
                'method': 'residual_learning'
            }

        except Exception as e:
            logging.error(f"Residual adjustment error: {e}")
            return {'adjustment': 0, 'method': 'error'}

    def _integrate_predictions(self, strategic: Dict, ml: Optional[Dict], residual: Dict) -> Dict[str, Any]:
        """예측 통합"""
        # ML 예측이 없는 경우
        if not ml:
            base_up = strategic['base_up']
            mi = strategic['mi']
            integration_method = 'strategic_only'
        else:
            # 가중 평균 계산
            base_up = (
                strategic['base_up'] * self.component_weights['strategic_rules'] +
                ml['base_up'] * self.component_weights['ml_validation']
            )
            mi = (
                strategic['mi'] * self.component_weights['strategic_rules'] +
                ml['mi'] * self.component_weights['ml_validation']
            )
            integration_method = 'weighted_average'

        # 잔차 조정 적용
        total_adjustment = residual['adjustment'] * self.component_weights['residual_learning']
        base_up += total_adjustment * 0.6
        mi += total_adjustment * 0.4

        # 범위 제한 (합리적 범위 내)
        base_up = max(2.0, min(4.5, round(base_up, 1)))
        mi = max(1.5, min(3.0, round(mi, 1)))

        return {
            'base_up': base_up,
            'mi': mi,
            'total': round(base_up + mi, 1),
            'method': integration_method,
            'weights_used': self.component_weights
        }

    def _calculate_confidence(self, strategic: Dict, ml: Optional[Dict]) -> Dict[str, Any]:
        """예측 신뢰도 계산"""
        confidence_factors = []

        # 1. 전략적 규칙 신뢰도 (breakdown 있으면 높음)
        if strategic.get('breakdown'):
            confidence_factors.append(('strategic_rules', 0.9))
        else:
            confidence_factors.append(('strategic_rules', 0.7))

        # 2. ML 모델 신뢰도
        if ml:
            # 전략적 예측과 ML 예측의 일치도
            diff = abs(strategic['total'] - ml['total'])
            if diff < 0.5:
                confidence_factors.append(('ml_agreement', 0.9))
            elif diff < 1.0:
                confidence_factors.append(('ml_agreement', 0.7))
            else:
                confidence_factors.append(('ml_agreement', 0.5))
        else:
            confidence_factors.append(('no_ml', 0.6))

        # 3. 데이터 품질
        if data_service.current_data is not None:
            data_size = len(data_service.current_data)
            if data_size >= 100:
                confidence_factors.append(('data_quality', 0.8))
            elif data_size >= 50:
                confidence_factors.append(('data_quality', 0.7))
            else:
                confidence_factors.append(('data_quality', 0.6))

        # 전체 신뢰도 계산
        overall_confidence = np.mean([score for _, score in confidence_factors])

        return {
            'overall': round(overall_confidence, 2),
            'factors': confidence_factors,
            'level': self._get_confidence_level(overall_confidence)
        }

    def _get_confidence_level(self, score: float) -> str:
        """신뢰도 수준 결정"""
        if score >= 0.8:
            return 'HIGH'
        elif score >= 0.6:
            return 'MEDIUM'
        else:
            return 'LOW'

    def _perform_sensitivity_analysis(self, prediction: Dict, params: Optional[Dict]) -> Dict[str, Any]:
        """민감도 분석"""
        sensitivity_results = {}
        base_total = prediction['total']

        # 주요 변수 목록
        key_variables = ['min_wage', 'gdp', 'revenue_growth', 'profit_margin', 'industry_avg']

        for var in key_variables:
            if params and var in params:
                original_value = params[var]
                impacts = []

                # ±20% 범위에서 테스트
                for pct_change in [-20, -10, 10, 20]:
                    test_params = params.copy()
                    test_params[var] = original_value * (1 + pct_change/100)

                    # 재예측
                    test_pred = self._get_strategic_prediction('custom', test_params)
                    impact = test_pred['total'] - base_total

                    impacts.append({
                        'change_pct': pct_change,
                        'new_value': test_params[var],
                        'impact': round(impact, 2)
                    })

                sensitivity_results[var] = {
                    'original': original_value,
                    'impacts': impacts,
                    'max_impact': max(abs(i['impact']) for i in impacts)
                }

        # 가장 민감한 변수 찾기
        if sensitivity_results:
            most_sensitive = max(sensitivity_results.items(),
                               key=lambda x: x[1]['max_impact'])
            return {
                'variables': sensitivity_results,
                'most_sensitive': most_sensitive[0],
                'max_impact': most_sensitive[1]['max_impact']
            }
        else:
            return {'variables': {}, 'most_sensitive': None, 'max_impact': 0}

    def _generate_recommendation(self, prediction: Dict, confidence: Dict) -> Dict[str, Any]:
        """최종 권고사항 생성"""
        base_up = prediction['base_up']
        mi = prediction['mi']
        total = prediction['total']
        conf_level = confidence['level']

        # 권고 메시지 생성
        if conf_level == 'HIGH':
            confidence_msg = "높은 신뢰도로 예측되었습니다."
        elif conf_level == 'MEDIUM':
            confidence_msg = "중간 신뢰도로 예측되었습니다. 추가 검토를 권장합니다."
        else:
            confidence_msg = "낮은 신뢰도입니다. 신중한 검토가 필요합니다."

        # 조정 권고
        adjustments = []
        if total > 5.5:
            adjustments.append("업계 평균 대비 높은 수준입니다. 성과 연동 강화를 고려하세요.")
        elif total < 4.5:
            adjustments.append("업계 평균 대비 낮은 수준입니다. 인재 이탈 리스크를 검토하세요.")

        return {
            'primary': f"Base-up {base_up}% + MI {mi}% = 총 {total}%",
            'confidence_message': confidence_msg,
            'adjustments': adjustments,
            'monitoring_points': [
                "분기별 경제지표 모니터링",
                "경쟁사 임금 동향 추적",
                "CDMO 수주 실적 반영"
            ],
            'risk_factors': self._identify_risk_factors(prediction)
        }

    def _identify_risk_factors(self, prediction: Dict) -> List[str]:
        """리스크 요인 식별"""
        risks = []
        total = prediction['total']

        if total > 6.0:
            risks.append("과도한 인건비 상승 리스크")
        if total < 4.0:
            risks.append("핵심 인재 이탈 리스크")

        # 추가 리스크 요인
        risks.extend([
            "글로벌 경기 침체 가능성",
            "CDMO 시장 경쟁 심화",
            "환율 변동성 확대"
        ])

        return risks[:3]  # 상위 3개만 반환

    def _prepare_scenario_data(self, scenario: str, custom_params: Optional[Dict]) -> pd.DataFrame:
        """시나리오 데이터 준비"""
        # 기본 시나리오 파라미터
        scenario_params = {
            'base': {
                'gdp': 1.6, 'min_wage': 3.5, 'cpi': 1.9,
                'revenue_growth': 15, 'profit_margin': 12,
                'unemployment': 3.5, 'exchange_rate': 1300
            },
            'conservative': {
                'gdp': 1.2, 'min_wage': 4.0, 'cpi': 2.5,
                'revenue_growth': 10, 'profit_margin': 10,
                'unemployment': 4.0, 'exchange_rate': 1400
            },
            'optimistic': {
                'gdp': 2.0, 'min_wage': 3.0, 'cpi': 1.8,
                'revenue_growth': 25, 'profit_margin': 15,
                'unemployment': 3.0, 'exchange_rate': 1200
            }
        }

        # 파라미터 선택
        if scenario == 'custom' and custom_params:
            params = custom_params
        else:
            params = scenario_params.get(scenario, scenario_params['base'])

        # DataFrame 생성
        return pd.DataFrame([params])

    def _analyze_historical_residuals(self) -> Dict[str, Any]:
        """과거 잔차 패턴 분석"""
        try:
            # 과거 데이터가 있는 경우
            if hasattr(strategic_dashboard_service, 'historical_data'):
                historical = strategic_dashboard_service.historical_data

                # 실제값과 예측값의 차이 계산 (간단한 예시)
                residuals = []
                for _, row in historical.iterrows():
                    # 여기서는 단순화를 위해 평균과의 차이로 계산
                    mean_total = historical['total'].mean()
                    residual = row['total'] - mean_total
                    residuals.append(residual)

                return {
                    'mean_residual': np.mean(residuals),
                    'std_residual': np.std(residuals),
                    'pattern': 'stable' if np.std(residuals) < 0.5 else 'volatile'
                }
            else:
                return {'pattern': 'no_data'}

        except Exception as e:
            logging.error(f"Historical residual analysis error: {e}")
            return {'pattern': 'error'}

    def extract_feature_importance(self) -> Dict[str, Any]:
        """ML 모델로부터 Feature Importance 추출"""
        try:
            importance_results = {}

            # GBR 모델이 있는 경우
            if modeling_service.validation_models.get('gbr'):
                gbr_model = modeling_service.validation_models['gbr']

                # Feature importance 추출
                if hasattr(gbr_model, 'feature_importances_'):
                    feature_names = modeling_service.feature_names or []
                    importances = gbr_model.feature_importances_

                    importance_df = pd.DataFrame({
                        'feature': feature_names,
                        'importance': importances
                    }).sort_values('importance', ascending=False)

                    importance_results['gbr'] = importance_df.to_dict('records')

            # Ridge 모델의 계수
            if modeling_service.validation_models.get('ridge'):
                ridge_model = modeling_service.validation_models['ridge']

                if hasattr(ridge_model, 'coef_'):
                    feature_names = modeling_service.feature_names or []
                    coefficients = ridge_model.coef_

                    coef_df = pd.DataFrame({
                        'feature': feature_names,
                        'coefficient': coefficients,
                        'abs_coefficient': np.abs(coefficients)
                    }).sort_values('abs_coefficient', ascending=False)

                    importance_results['ridge'] = coef_df.to_dict('records')

            # 전략적 규칙의 가중치와 비교
            strategic_weights = {}
            for rule_name, rule in strategic_dashboard_service.decision_rules.items():
                strategic_weights[rule_name] = rule['weight']

            importance_results['strategic'] = strategic_weights

            # 통합 중요도 계산
            self.integrated_importance = self._integrate_importance(importance_results)

            return {
                'ml_importance': importance_results,
                'integrated': self.integrated_importance,
                'top_features': self._get_top_features(self.integrated_importance)
            }

        except Exception as e:
            logging.error(f"Feature importance extraction error: {e}")
            return {'error': str(e)}

    def _integrate_importance(self, importance_results: Dict) -> Dict[str, float]:
        """다양한 소스의 중요도 통합"""
        integrated = {}

        # GBR importance
        if 'gbr' in importance_results:
            for item in importance_results['gbr']:
                integrated[item['feature']] = integrated.get(item['feature'], 0) + item['importance'] * 0.3

        # Ridge coefficients
        if 'ridge' in importance_results:
            for item in importance_results['ridge']:
                integrated[item['feature']] = integrated.get(item['feature'], 0) + item['abs_coefficient'] * 0.2

        # Strategic weights
        if 'strategic' in importance_results:
            for rule, weight in importance_results['strategic'].items():
                integrated[rule] = integrated.get(rule, 0) + weight * 0.5

        # 정규화
        total = sum(integrated.values())
        if total > 0:
            integrated = {k: v/total for k, v in integrated.items()}

        return integrated

    def _get_top_features(self, integrated_importance: Dict, top_n: int = 5) -> List[Dict]:
        """상위 중요 특성 추출"""
        sorted_features = sorted(integrated_importance.items(), key=lambda x: x[1], reverse=True)

        return [
            {'feature': feature, 'importance': round(importance, 3)}
            for feature, importance in sorted_features[:top_n]
        ]

    def validate_prediction_quality(self) -> Dict[str, Any]:
        """예측 품질 검증"""
        validation_results = {
            'strategic_validation': self._validate_strategic_rules(),
            'ml_validation': self._validate_ml_models(),
            'historical_validation': self._validate_against_history(),
            'overall_quality': None
        }

        # 전체 품질 점수 계산
        scores = []
        for key, result in validation_results.items():
            if key != 'overall_quality' and isinstance(result, dict) and 'score' in result:
                scores.append(result['score'])

        validation_results['overall_quality'] = {
            'score': np.mean(scores) if scores else 0,
            'level': self._get_quality_level(np.mean(scores) if scores else 0)
        }

        return validation_results

    def _validate_strategic_rules(self) -> Dict[str, Any]:
        """전략적 규칙 검증"""
        try:
            # 각 시나리오에 대한 예측 테스트
            scenarios = ['base', 'conservative', 'optimistic']
            predictions = []

            for scenario in scenarios:
                pred = self._get_strategic_prediction(scenario, None)
                predictions.append(pred['total'])

            # 예측의 합리성 검증
            is_reasonable = (
                min(predictions) >= 3.0 and  # 최소 3%
                max(predictions) <= 7.0 and  # 최대 7%
                predictions[1] < predictions[0] < predictions[2]  # 보수 < 기본 < 낙관
            )

            return {
                'score': 0.9 if is_reasonable else 0.6,
                'predictions': predictions,
                'is_reasonable': is_reasonable
            }

        except Exception as e:
            return {'score': 0.5, 'error': str(e)}

    def _validate_ml_models(self) -> Dict[str, Any]:
        """ML 모델 검증"""
        try:
            if not modeling_service.current_model:
                return {'score': 0.5, 'message': 'No ML model available'}

            # 모델 성능 메트릭 확인
            if hasattr(modeling_service, 'model_results'):
                metrics = modeling_service.model_results
                r2_score = metrics.get('r2', 0)

                # R2 점수 기반 품질 평가
                if r2_score > 0.7:
                    score = 0.9
                elif r2_score > 0.5:
                    score = 0.7
                else:
                    score = 0.5

                return {
                    'score': score,
                    'r2': r2_score,
                    'metrics': metrics
                }
            else:
                return {'score': 0.6, 'message': 'No performance metrics available'}

        except Exception as e:
            return {'score': 0.5, 'error': str(e)}

    def _validate_against_history(self) -> Dict[str, Any]:
        """과거 데이터 대비 검증"""
        try:
            historical = strategic_dashboard_service.historical_data
            historical_mean = historical['total'].mean()
            historical_std = historical['total'].std()

            # 현재 예측
            current_pred = self.predict_wage_increase(2026, 'base')
            pred_total = current_pred['prediction']['total']

            # 과거 범위 내에 있는지 확인
            z_score = abs((pred_total - historical_mean) / historical_std)

            if z_score < 1:  # 1 표준편차 이내
                score = 0.9
            elif z_score < 2:  # 2 표준편차 이내
                score = 0.7
            else:
                score = 0.5

            return {
                'score': score,
                'historical_mean': round(historical_mean, 2),
                'historical_std': round(historical_std, 2),
                'prediction': pred_total,
                'z_score': round(z_score, 2)
            }

        except Exception as e:
            return {'score': 0.5, 'error': str(e)}

    def _get_quality_level(self, score: float) -> str:
        """품질 수준 결정"""
        if score >= 0.8:
            return 'EXCELLENT'
        elif score >= 0.7:
            return 'GOOD'
        elif score >= 0.6:
            return 'FAIR'
        else:
            return 'POOR'


# 싱글톤 인스턴스
hybrid_prediction_service = HybridPredictionService()